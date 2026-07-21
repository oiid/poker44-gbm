"""Telemetry feature bank for Poker44 v3.0 subject sessions (STAGING).

Target payload: ``contracts/subject-session.v1.schema.json`` on
``origin/dev`` of Poker44-subnet, i.e. the v3.0 "subject session" that
replaces the old chunk format:

    {
      "schema_version": "1",
      "session_id": str, "window_id": str,
      "hands": [ {"hand_number": int, "actions": [ {...} ]} ],      # 1..512
      "telemetry": {
        "events": [ {"sequence": int, "offset_ms": int,
                     "source": "client"|"server",
                     "event_type": str, "target": str|None,
                     "value": <any JSON>} ],                        # <=50000
        "summary": {"event_count","action_count","duration_ms",
                    "decision_count","decision_mean_ms","decision_std_ms"}
      }
    }

The v3 payload deleted almost everything the old 468-feature chunk
extractor fed on (metadata, players roster, streets/board, outcome /
showdown / payouts). What it ADDED is timing and interaction telemetry:
``decision_time_ms``, ``time_since_last_action_ms``, ``occurred_at`` and
the whole ``telemetry`` block. This module is the greenfield feature
bank for exactly that new surface.

Design invariants
-----------------
* **Never raises.** Every group is executed inside a guard; a group that
  throws (or returns the wrong arity) is replaced by zeros. A totally
  malformed session -> all-zero vector with presence flags at 0.
* **Fixed arity.** Every emit is unconditional, so ``FEATURE_NAMES`` is
  derived at import time by running the extractor on ``{}``. The vector
  length never depends on the input.
* **Size invariant.** Nothing is a raw count: counts become rates
  (per minute / per action / per hand) or shares. Distribution blocks are
  quantiles and shape statistics.
* **Hand-order invariant where appropriate.** Groups 01-09 are invariant
  to any permutation of the hands (they sort or aggregate). Group 10
  (``drift``) is deliberately order-DEPENDENT: non-stationarity over the
  session is the signal, so it consumes hands in ``hand_number`` order
  (falling back to array order).
* **stdlib + numpy only.**

Public API
----------
``extract_session_features(session) -> (names, values)``
``extract_batch(sessions) -> (names, np.ndarray[n, d])``
``FEATURE_NAMES``, ``FEATURE_GROUPS`` (name -> feature names),
``FEATURE_GROUP_INDICES`` (name -> column indices), ``N_FEATURES``.

Feature groups (hypotheses documented in README_telemetry_features.md):
  g01_presence     data-completeness / missingness fingerprint
  g02_summary      the 6 summary scalars + summary-vs-observed consistency
  g03_interact     inter-event timing distribution (telemetry stream)
  g04_decision     poker think-time distribution (decision_time_ms, tsla)
  g05_regular      quantization / autocorrelation / spectrum / entropy
  g06_coupling     think time vs decision complexity (the poker-specific edge)
  g07_motor        pointer/click motor signature recovered from event.value
  g08_attention    focus/blur, idle gaps, burstiness, fatigue drift
  g09_sequence     n-gram entropy / compressibility / repeated lines
  g10_drift        cross-hand stationarity (humans drift, bots do not)
"""

from __future__ import annotations

import math
import zlib
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

__all__ = [
    "extract_session_features",
    "extract_batch",
    "FEATURE_NAMES",
    "FEATURE_GROUPS",
    "FEATURE_GROUP_INDICES",
    "N_FEATURES",
    "GROUP_ORDER",
]

_EPS = 1e-9
_STREETS = ("preflop", "flop", "turn", "river")
_ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
_AGGRESSIVE = ("bet", "raise", "all_in")
_PASSIVE = ("check", "call")

# Timing grids (ms) probed for quantization. Chosen to cover setTimeout /
# frame-tick / poll-loop granularities that automation harnesses produce.
_GRIDS = (10, 16, 20, 25, 50, 100, 125, 250, 500, 1000)

# Log-spaced decision-time buckets used for sequence tokenisation (ms).
_TIME_BUCKETS = (150.0, 350.0, 700.0, 1400.0, 3000.0, 6000.0, 15000.0)

# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------


def _num(value: Any, default: float | None = None) -> float | None:
    """Best-effort numeric coercion. Booleans are NOT numbers here."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.strip())
        except (TypeError, ValueError):
            return default
    else:
        return default
    if not math.isfinite(f):
        return default
    return f


def _clean(x: Any) -> float:
    """Coerce to a finite, bounded float (feature values must be safe)."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    if f > 1e9:
        return 1e9
    if f < -1e9:
        return -1e9
    return f


def _safe_div(a: float, b: float) -> float:
    if abs(b) < _EPS:
        return 0.0
    return a / b


def _rel(a: float, b: float) -> float:
    """a / (|b| + eps): a scale-free ratio that never blows up."""
    return a / (abs(b) + _EPS)


def _log10p(a: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(a, 0.0, None) + 1.0)


def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return 0.0
    xs, ys = x.std(), y.std()
    if xs < _EPS or ys < _EPS:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (xs * ys))


def _rank(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, float)
    ranks[order] = np.arange(a.size, dtype=float)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return 0.0
    return _pearson(_rank(x), _rank(y))


def _gini(a: np.ndarray) -> float:
    a = _finite(a)
    if a.size < 2:
        return 0.0
    a = np.sort(np.clip(a, 0.0, None))
    s = a.sum()
    if s < _EPS:
        return 0.0
    n = a.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (idx * a).sum()) / (n * s) - (n + 1.0) / n)


def _entropy_counts(counts) -> float:
    """Shannon entropy in nats of a count iterable."""
    vals = [float(c) for c in counts if c > 0]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log(c / total) for c in vals))


def _norm_entropy(counts) -> float:
    """Entropy normalised by log(k) -> [0, 1]."""
    vals = [float(c) for c in counts if c > 0]
    if len(vals) < 2:
        return 0.0
    return float(_entropy_counts(vals) / math.log(len(vals)))


def _acf(x: np.ndarray, lag: int) -> float:
    x = _finite(x)
    if x.size < lag + 5:
        return 0.0
    return _pearson(x[:-lag], x[lag:])


def _runs_z(x: np.ndarray) -> float:
    """Wald-Wolfowitz runs-test z for above/below median."""
    x = _finite(x)
    if x.size < 12:
        return 0.0
    med = np.median(x)
    b = x > med
    n1 = int(b.sum())
    n0 = int(x.size - n1)
    if n0 < 2 or n1 < 2:
        return 0.0
    runs = 1 + int((b[1:] != b[:-1]).sum())
    n = n0 + n1
    mu = 2.0 * n0 * n1 / n + 1.0
    var = (mu - 1.0) * (mu - 2.0) / max(n - 1.0, 1.0)
    if var < _EPS:
        return 0.0
    return float((runs - mu) / math.sqrt(var))


def _spectral(x: np.ndarray) -> tuple[float, float, float, float]:
    """(peak_share, top3_share, spectral_entropy_norm, spectral_flatness)."""
    x = _finite(x)
    if x.size < 16:
        return 0.0, 0.0, 0.0, 0.0
    if x.size > 4096:
        x = x[:4096]
    x = x - x.mean()
    sd = x.std()
    if sd < _EPS:
        return 1.0, 1.0, 0.0, 0.0
    x = x / sd
    psd = np.abs(np.fft.rfft(x)) ** 2
    psd = psd[1:]  # drop DC
    total = psd.sum()
    if total < _EPS or psd.size < 3:
        return 0.0, 0.0, 0.0, 0.0
    p = psd / total
    peak = float(p.max())
    top3 = float(np.sort(p)[-3:].sum())
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum() / math.log(p.size))
    gm = float(np.exp(np.log(psd + _EPS).mean()))
    flat = float(gm / (psd.mean() + _EPS))
    return peak, top3, ent, flat


def _perm_entropy(x: np.ndarray, order: int = 3) -> float:
    """Bandt-Pompe permutation entropy, normalised."""
    x = _finite(x)
    if x.size < order + 10:
        return 0.0
    if x.size > 5000:
        x = x[:5000]
    win = np.lib.stride_tricks.sliding_window_view(x, order)
    pat = np.argsort(win, axis=1, kind="mergesort")
    weights = (order ** np.arange(order)).astype(np.int64)
    codes = (pat * weights).sum(axis=1)
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    if counts.size < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(order)))


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    x = _finite(x)
    if x.size < 24:
        return 0.0
    if x.size > 500:
        x = x[np.linspace(0, x.size - 1, 500).astype(int)]
    sd = float(x.std())
    if sd < _EPS:
        return 0.0
    tol = r * sd

    def _count(mm: int) -> float:
        n = x.size - mm + 1
        if n < 3:
            return 0.0
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        np.fill_diagonal(d, np.inf)
        return float((d <= tol).sum())

    b = _count(m)
    a = _count(m + 1)
    if b <= 0.0 or a <= 0.0:
        return 4.0  # capped "maximally irregular"
    return float(min(4.0, -math.log(a / b)))


def _lz_complexity(tokens: list[str]) -> float:
    """Lempel-Ziv-76 complexity, normalised by n/log(n)."""
    if len(tokens) < 8:
        return 0.0
    seq = tokens[:3000]
    n = len(seq)
    seen: set[tuple] = set()
    i = 0
    c = 0
    while i < n:
        j = 1
        while i + j <= n and tuple(seq[i : i + j]) in seen:
            j += 1
        seen.add(tuple(seq[i : i + j]))
        c += 1
        i += j
    norm = n / max(math.log(max(n, 2), 2), _EPS)
    return float(c / max(norm, _EPS))


def _zlib_ratio(tokens: list[str]) -> float:
    """Compression ratio of the token stream mapped to single bytes."""
    if len(tokens) < 8:
        return 0.0
    seq = tokens[:4000]
    vocab: dict[str, int] = {}
    raw = bytearray()
    for t in seq:
        if t not in vocab:
            vocab[t] = len(vocab) % 251
        raw.append(vocab[t])
    blob = bytes(raw)
    comp = zlib.compress(blob, 6)
    return float(len(comp) / max(len(blob), 1))


def _longest_repeat_share(tokens: list[str]) -> float:
    """Longest substring occurring at least twice, as a share of length."""
    n = len(tokens)
    if n < 8:
        return 0.0
    seq = tokens[:2000]
    n = len(seq)

    def has_repeat(k: int) -> bool:
        seen: set[tuple] = set()
        for i in range(n - k + 1):
            w = tuple(seq[i : i + k])
            if w in seen:
                return True
            seen.add(w)
        return False

    lo, hi = 1, min(n // 2, 200)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_repeat(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return float(best / n)


def _max_run_share(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    best = cur = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return float(best / len(tokens))


def _parse_iso(value: Any) -> float | None:
    """Parse an RFC3339 timestamp into a POSIX float. Never raises."""
    if not isinstance(value, str) or not value:
        return None
    s = value.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
    try:
        return dt.timestamp()
    except (OverflowError, OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# emission plumbing
# ---------------------------------------------------------------------------


class _Out:
    """Ordered (name, value) accumulator for a single feature group."""

    __slots__ = ("prefix", "names", "values")

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.names: list[str] = []
        self.values: list[float] = []

    def add(self, name: str, value: Any) -> None:
        self.names.append(self.prefix + "__" + name)
        self.values.append(_clean(value))

    def addm(self, pairs) -> None:
        for name, value in pairs:
            self.add(name, value)


_DIST_SUFFIXES = (
    "present",
    "mean",
    "std",
    "cv",
    "mad_rel",
    "min",
    "max",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "iqr_rel",
    "skew",
    "kurt",
    "p90_p50",
    "p50_p10",
    "gini",
)
_DIST_N = len(_DIST_SUFFIXES)


def _dist(out: _Out, prefix: str, values, min_n: int = 3) -> None:
    """Emit the fixed 20-stat distribution block for ``values``."""
    a = _finite(np.asarray(list(values), dtype=float) if not isinstance(values, np.ndarray) else values)
    if a.size < min_n:
        for suf in _DIST_SUFFIXES:
            out.add(f"{prefix}_{suf}", 0.0)
        return
    mean = float(a.mean())
    std = float(a.std())
    qs = np.quantile(a, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    q05, q10, q25, q50, q75, q90, q95 = (float(v) for v in qs)
    mad = float(np.abs(a - q50).mean())
    if std > _EPS:
        z = (a - mean) / std
        skew = float((z**3).mean())
        kurt = float((z**4).mean() - 3.0)
    else:
        skew = 0.0
        kurt = 0.0
    out.add(f"{prefix}_present", 1.0)
    out.add(f"{prefix}_mean", mean)
    out.add(f"{prefix}_std", std)
    out.add(f"{prefix}_cv", _rel(std, mean))
    out.add(f"{prefix}_mad_rel", _rel(mad, q50))
    out.add(f"{prefix}_min", float(a.min()))
    out.add(f"{prefix}_max", float(a.max()))
    out.add(f"{prefix}_q05", q05)
    out.add(f"{prefix}_q10", q10)
    out.add(f"{prefix}_q25", q25)
    out.add(f"{prefix}_q50", q50)
    out.add(f"{prefix}_q75", q75)
    out.add(f"{prefix}_q90", q90)
    out.add(f"{prefix}_q95", q95)
    out.add(f"{prefix}_iqr_rel", _rel(q75 - q25, q50))
    out.add(f"{prefix}_skew", skew)
    out.add(f"{prefix}_kurt", kurt)
    out.add(f"{prefix}_p90_p50", _rel(q90, q50))
    out.add(f"{prefix}_p50_p10", _rel(q50, q10))
    out.add(f"{prefix}_gini", _gini(a))


_SEQ_SUFFIXES = (
    "present",
    "unique_rate",
    "top1_share",
    "top3_share",
    "h1_norm",
    "h2_norm",
    "h3_norm",
    "cond_h_norm",
    "bigram_unique_rate",
    "repeat_pair_rate",
    "lz_norm",
    "zlib_ratio",
    "longest_repeat_share",
    "max_run_share",
    "determinism",
)
_SEQ_N = len(_SEQ_SUFFIXES)


def _seq_block(out: _Out, prefix: str, tokens: list[str]) -> None:
    """Emit the fixed 15-stat sequence-regularity block."""
    n = len(tokens)
    if n < 6:
        for suf in _SEQ_SUFFIXES:
            out.add(f"{prefix}_{suf}", 0.0)
        return
    uni = Counter(tokens)
    bi = Counter(zip(tokens, tokens[1:]))
    tri = Counter(zip(tokens, tokens[1:], tokens[2:]))
    top = uni.most_common(3)
    top1 = top[0][1] / n
    top3 = sum(c for _, c in top) / n
    h1 = _entropy_counts(uni.values())
    h2 = _entropy_counts(bi.values())
    # H(x_{t+1} | x_t) = H2 - H1 (in nats), normalised by H1
    cond = max(0.0, h2 - h1)
    # transition determinism: mean over states of max outgoing probability
    outgoing: dict[str, Counter] = {}
    for a, b in zip(tokens, tokens[1:]):
        outgoing.setdefault(a, Counter())[b] += 1
    det = (
        float(np.mean([max(c.values()) / sum(c.values()) for c in outgoing.values()]))
        if outgoing
        else 0.0
    )
    out.add(f"{prefix}_present", 1.0)
    out.add(f"{prefix}_unique_rate", len(uni) / n)
    out.add(f"{prefix}_top1_share", top1)
    out.add(f"{prefix}_top3_share", top3)
    out.add(f"{prefix}_h1_norm", _norm_entropy(uni.values()))
    out.add(f"{prefix}_h2_norm", _norm_entropy(bi.values()))
    out.add(f"{prefix}_h3_norm", _norm_entropy(tri.values()))
    out.add(f"{prefix}_cond_h_norm", _rel(cond, h1))
    out.add(f"{prefix}_bigram_unique_rate", len(bi) / max(n - 1, 1))
    out.add(f"{prefix}_repeat_pair_rate", sum(1 for a, b in zip(tokens, tokens[1:]) if a == b) / max(n - 1, 1))
    out.add(f"{prefix}_lz_norm", _lz_complexity(tokens))
    out.add(f"{prefix}_zlib_ratio", _zlib_ratio(tokens))
    out.add(f"{prefix}_longest_repeat_share", _longest_repeat_share(tokens))
    out.add(f"{prefix}_max_run_share", _max_run_share(tokens))
    out.add(f"{prefix}_determinism", det)


_GRID_SUFFIXES = tuple(f"grid{g}_share" for g in _GRIDS) + (
    "best_grid_log2",
    "best_grid_share",
    "best_grid_lift",
    "gcd_log2",
    "gcd_gt1",
    "last_digit_h",
    "last2_digit_h",
    "unique_rate",
    "top_value_share",
    "int_share",
)
_GRID_N = len(_GRID_SUFFIXES)


def _grid_block(out: _Out, prefix: str, values) -> None:
    """Quantisation / tick-boundary detection on a millisecond series."""
    a = _finite(np.asarray(list(values), dtype=float) if not isinstance(values, np.ndarray) else values)
    a = a[a >= 0]
    if a.size < 8:
        for suf in _GRID_SUFFIXES:
            out.add(f"{prefix}_{suf}", 0.0)
        return
    ints = np.rint(a).astype(np.int64)
    int_share = float(np.mean(np.abs(a - ints) < 1e-6))
    nz = ints[ints > 0]
    best_g, best_share, best_lift = 0, 0.0, 0.0
    for g in _GRIDS:
        share = float(np.mean(ints % g == 0))
        out.add(f"{prefix}_grid{g}_share", share)
        lift = math.log2(max(share, _EPS) * g + _EPS)
        if lift > best_lift:
            best_g, best_share, best_lift = g, share, lift
    gcd = 0
    if nz.size:
        gcd = int(np.gcd.reduce(nz))
    out.add(f"{prefix}_best_grid_log2", math.log2(best_g) if best_g else 0.0)
    out.add(f"{prefix}_best_grid_share", best_share)
    out.add(f"{prefix}_best_grid_lift", best_lift)
    out.add(f"{prefix}_gcd_log2", math.log2(gcd) if gcd > 0 else 0.0)
    out.add(f"{prefix}_gcd_gt1", 1.0 if gcd > 1 else 0.0)
    out.add(f"{prefix}_last_digit_h", _norm_entropy(Counter((ints % 10).tolist()).values()))
    out.add(f"{prefix}_last2_digit_h", _norm_entropy(Counter((ints % 100).tolist()).values()))
    counts = Counter(ints.tolist())
    out.add(f"{prefix}_unique_rate", len(counts) / ints.size)
    out.add(f"{prefix}_top_value_share", max(counts.values()) / ints.size)
    out.add(f"{prefix}_int_share", int_share)


_REG_SUFFIXES = (
    "acf1",
    "acf2",
    "acf3",
    "acf5",
    "acf10",
    "acf_absmax",
    "acf_absmean",
    "runs_z",
    "spec_peak",
    "spec_top3",
    "spec_ent",
    "spec_flat",
    "perm_ent",
    "samp_ent",
    "block_std_cv",
    "block_std_ratio",
    "block_mean_cv",
    "vov",
)
_REG_N = len(_REG_SUFFIXES)


def _reg_block(out: _Out, prefix: str, values) -> None:
    """Serial-structure block: autocorrelation, spectrum, entropies, vov."""
    a = _finite(np.asarray(list(values), dtype=float) if not isinstance(values, np.ndarray) else values)
    if a.size < 12:
        for suf in _REG_SUFFIXES:
            out.add(f"{prefix}_{suf}", 0.0)
        return
    acfs = [_acf(a, lag) for lag in (1, 2, 3, 5, 10)]
    all_acf = [abs(_acf(a, lag)) for lag in range(1, 11)]
    peak, top3, ent, flat = _spectral(a)
    # variance-of-variance over contiguous blocks
    k = max(2, min(8, a.size // 8))
    blocks = np.array_split(a[: (a.size // k) * k], k)
    b_std = np.array([b.std() for b in blocks if b.size > 1], dtype=float)
    b_mean = np.array([b.mean() for b in blocks if b.size > 1], dtype=float)
    if b_std.size >= 2:
        std_cv = _rel(float(b_std.std()), float(b_std.mean()))
        std_ratio = _rel(float(b_std.max()), float(b_std.min()) + 1e-6)
        mean_cv = _rel(float(b_mean.std()), float(b_mean.mean()))
        vov = _rel(float(np.var(b_std)), float(np.var(a)))
    else:
        std_cv = std_ratio = mean_cv = vov = 0.0
    out.add(f"{prefix}_acf1", acfs[0])
    out.add(f"{prefix}_acf2", acfs[1])
    out.add(f"{prefix}_acf3", acfs[2])
    out.add(f"{prefix}_acf5", acfs[3])
    out.add(f"{prefix}_acf10", acfs[4])
    out.add(f"{prefix}_acf_absmax", max(all_acf) if all_acf else 0.0)
    out.add(f"{prefix}_acf_absmean", float(np.mean(all_acf)) if all_acf else 0.0)
    out.add(f"{prefix}_runs_z", _runs_z(a))
    out.add(f"{prefix}_spec_peak", peak)
    out.add(f"{prefix}_spec_top3", top3)
    out.add(f"{prefix}_spec_ent", ent)
    out.add(f"{prefix}_spec_flat", flat)
    out.add(f"{prefix}_perm_ent", _perm_entropy(a, 3))
    out.add(f"{prefix}_samp_ent", _sample_entropy(a))
    out.add(f"{prefix}_block_std_cv", std_cv)
    out.add(f"{prefix}_block_std_ratio", std_ratio)
    out.add(f"{prefix}_block_mean_cv", mean_cv)
    out.add(f"{prefix}_vov", vov)


# ---------------------------------------------------------------------------
# session context
# ---------------------------------------------------------------------------

_COORD_KEYS = (
    ("x", "y"),
    ("clientx", "clienty"),
    ("pagex", "pagey"),
    ("offsetx", "offsety"),
    ("screenx", "screeny"),
    ("px", "py"),
    ("cx", "cy"),
    ("left", "top"),
    ("col", "row"),
)
_COORD_CONTAINERS = ("position", "pos", "point", "coords", "coordinate", "coordinates", "xy", "at")

_CLICKY = ("click", "tap", "press", "down", "up", "select", "button", "submit")
_MOVEY = ("move", "drag", "hover", "pointermove", "mousemove", "swipe")

_ATTENTION_FAMILIES: dict[str, tuple[str, ...]] = {
    "focus": ("focus", "active", "resume", "visible"),
    "blur": ("blur", "hidden", "visibility", "away", "background", "inactive", "unfocus"),
    "idle": ("idle", "timeout", "afk", "stall", "sit_out", "sitout"),
    "pointer": ("pointer", "mouse", "click", "tap", "touch", "press"),
    "move": ("move", "drag", "hover", "swipe"),
    "key": ("key", "type", "input", "text", "paste", "copy"),
    "scroll": ("scroll", "wheel", "zoom", "resize"),
    "nav": ("nav", "route", "page", "load", "open", "close", "enter", "leave", "join"),
    "bet_ui": ("slider", "bet", "raise", "amount", "chip", "stack", "pot"),
    "chat": ("chat", "message", "emote", "note", "avatar"),
}
_ATTENTION_KEYS = tuple(_ATTENTION_FAMILIES.keys())


def _walk_for_point(node: Any, depth: int = 0, budget: list | None = None):
    """Recursively look for a coordinate-like pair inside a telemetry value."""
    if budget is None:
        budget = [200]
    if depth > 4 or budget[0] <= 0:
        return None
    budget[0] -= 1
    if isinstance(node, dict):
        low = {str(k).lower(): v for k, v in node.items()}
        for kx, ky in _COORD_KEYS:
            if kx in low and ky in low:
                x = _num(low[kx])
                y = _num(low[ky])
                if x is not None and y is not None:
                    return (x, y)
        for key in _COORD_CONTAINERS:
            if key in low:
                found = _walk_for_point(low[key], depth + 1, budget)
                if found is not None:
                    return found
        for value in node.values():
            if isinstance(value, (dict, list, tuple)):
                found = _walk_for_point(value, depth + 1, budget)
                if found is not None:
                    return found
        return None
    if isinstance(node, (list, tuple)):
        if len(node) == 2:
            x = _num(node[0])
            y = _num(node[1])
            if x is not None and y is not None:
                return (x, y)
        for value in node:
            if isinstance(value, (dict, list, tuple)):
                found = _walk_for_point(value, depth + 1, budget)
                if found is not None:
                    return found
    return None


def _street_index(phase: Any) -> float:
    if not isinstance(phase, str):
        return -1.0
    p = phase.strip().lower()
    for i, name in enumerate(_STREETS):
        if p.startswith(name[:4]):
            return float(i)
    if p in ("pre", "p"):
        return 0.0
    return -1.0


class _Ctx:
    """Normalised, defensive view of one subject session."""

    def __init__(self, session: Any) -> None:
        self.ok = isinstance(session, dict)
        session = session if self.ok else {}

        tel = session.get("telemetry")
        self.has_tel = isinstance(tel, dict)
        tel = tel if self.has_tel else {}

        summary = tel.get("summary")
        self.has_summary = isinstance(summary, dict)
        self.summary = summary if self.has_summary else {}

        raw_events = tel.get("events")
        self.has_events_field = isinstance(raw_events, list)
        events = [e for e in (raw_events or []) if isinstance(e, dict)] if self.has_events_field else []
        # order by offset then declared sequence; both are schema-guaranteed ints
        events.sort(key=lambda e: (_num(e.get("offset_ms"), 0.0) or 0.0, _num(e.get("sequence"), 0.0) or 0.0))
        self.events = events
        self.n_events = len(events)

        self.ev_off = np.array([_num(e.get("offset_ms"), 0.0) or 0.0 for e in events], dtype=float)
        self.ev_type = [str(e.get("event_type") or "").strip().lower() for e in events]
        self.ev_target = [("" if e.get("target") is None else str(e.get("target")).strip().lower()) for e in events]
        self.ev_source = [str(e.get("source") or "").strip().lower() for e in events]
        self.ev_has_target = np.array([e.get("target") not in (None, "") for e in events], dtype=bool)
        self.ev_has_value = np.array([e.get("value") is not None for e in events], dtype=bool)
        self.ev_dt = np.diff(self.ev_off) if self.n_events >= 2 else np.zeros(0, dtype=float)
        self.ev_dt = self.ev_dt[self.ev_dt >= 0]

        # ---- hands / actions -------------------------------------------------
        raw_hands = session.get("hands")
        hands = [h for h in (raw_hands or []) if isinstance(h, dict)] if isinstance(raw_hands, list) else []
        indexed = list(enumerate(hands))
        indexed.sort(key=lambda ih: (_num(ih[1].get("hand_number"), float(ih[0])), ih[0]))
        self.hands = [h for _, h in indexed]
        self.n_hands = len(self.hands)

        self.hand_actions: list[list[dict]] = []
        for hand in self.hands:
            acts = hand.get("actions")
            acts = [a for a in (acts or []) if isinstance(a, dict)] if isinstance(acts, list) else []
            acts.sort(key=lambda a: (_num(a.get("sequence"), 0.0) or 0.0))
            self.hand_actions.append(acts)
        self.actions = [a for acts in self.hand_actions for a in acts]
        self.n_actions = len(self.actions)

        n = self.n_actions
        nan = float("nan")
        self.a_dec = np.full(n, nan)
        self.a_tsla = np.full(n, nan)
        self.a_pot = np.full(n, nan)
        self.a_call = np.full(n, nan)
        self.a_curbet = np.full(n, nan)
        self.a_stack = np.full(n, nan)
        self.a_active = np.full(n, nan)
        self.a_amount = np.full(n, nan)
        self.a_street = np.full(n, nan)
        self.a_idx_norm = np.full(n, nan)
        self.a_hand = np.zeros(n, dtype=int)
        self.a_occ = np.full(n, nan)
        self.a_type: list[str] = []
        self.a_evtype: list[str] = []
        self.a_pos: list[str] = []
        self.a_board = np.full(n, nan)
        self.a_allin = np.zeros(n, dtype=float)
        self.a_hole = np.zeros(n, dtype=float)

        k = 0
        for hi, acts in enumerate(self.hand_actions):
            m = max(len(acts), 1)
            for ai, act in enumerate(acts):
                self.a_dec[k] = _num(act.get("decision_time_ms"), nan)
                self.a_tsla[k] = _num(act.get("time_since_last_action_ms"), nan)
                self.a_pot[k] = _num(act.get("pot_size"), nan)
                self.a_call[k] = _num(act.get("call_amount"), nan)
                self.a_curbet[k] = _num(act.get("current_bet"), nan)
                self.a_stack[k] = _num(act.get("player_stack"), nan)
                self.a_active[k] = _num(act.get("active_players"), nan)
                self.a_amount[k] = _num(act.get("amount"), nan)
                self.a_street[k] = _street_index(act.get("phase"))
                self.a_idx_norm[k] = ai / m
                self.a_hand[k] = hi
                occ = _parse_iso(act.get("occurred_at"))
                self.a_occ[k] = nan if occ is None else occ
                self.a_type.append(str(act.get("action_type") or "").strip().lower())
                self.a_evtype.append(str(act.get("event_type") or "").strip().lower())
                pos = act.get("position_name")
                self.a_pos.append("" if pos is None else str(pos).strip().lower())
                board = act.get("community_cards")
                self.a_board[k] = float(len(board)) if isinstance(board, list) else nan
                self.a_allin[k] = 1.0 if act.get("is_all_in") is True else 0.0
                hole = act.get("hole_cards")
                self.a_hole[k] = 1.0 if isinstance(hole, list) and hole else 0.0
                k += 1

        self.a_street[self.a_street < 0] = nan
        self.dec = self.a_dec[np.isfinite(self.a_dec)]
        self.tsla = self.a_tsla[np.isfinite(self.a_tsla)]
        self.log_dec = _log10p(self.dec)
        self.log_dec_all = _log10p(np.where(np.isfinite(self.a_dec), self.a_dec, np.nan))

        # session duration: prefer summary, else telemetry span, else action span
        dur = _num(self.summary.get("duration_ms"), None)
        if dur is None or dur <= 0:
            dur = float(self.ev_off.max() - self.ev_off.min()) if self.n_events >= 2 else 0.0
        if dur <= 0:
            tot = np.nansum(np.where(np.isfinite(self.a_dec), self.a_dec, 0.0)) + np.nansum(
                np.where(np.isfinite(self.a_tsla), self.a_tsla, 0.0)
            )
            dur = float(tot)
        self.duration_ms = max(float(dur), 0.0)
        self.minutes = max(self.duration_ms / 60000.0, _EPS)

        # pointer / motor stream
        self.points: list[tuple[float, float]] = []
        self.point_dt: list[float] = []
        self.click_idx: list[int] = []
        for i, ev in enumerate(self.events):
            et = self.ev_type[i]
            if any(tok in et for tok in _CLICKY):
                self.click_idx.append(i)
            pt = _walk_for_point(ev.get("value"))
            if pt is not None:
                self.points.append(pt)
                self.point_dt.append(self.ev_off[i])


# ---------------------------------------------------------------------------
# g01 presence
# ---------------------------------------------------------------------------


def _g01_presence(ctx: _Ctx, out: _Out) -> None:
    n_ev = max(ctx.n_events, 1)
    n_ac = max(ctx.n_actions, 1)
    out.add("session_is_dict", 1.0 if ctx.ok else 0.0)
    out.add("has_telemetry", 1.0 if ctx.has_tel else 0.0)
    out.add("has_summary", 1.0 if ctx.has_summary else 0.0)
    out.add("has_events_field", 1.0 if ctx.has_events_field else 0.0)
    out.add("has_events", 1.0 if ctx.n_events > 0 else 0.0)
    out.add("has_hands", 1.0 if ctx.n_hands > 0 else 0.0)
    out.add("has_actions", 1.0 if ctx.n_actions > 0 else 0.0)
    out.add("has_client_events", 1.0 if any(s == "client" for s in ctx.ev_source) else 0.0)
    out.add("has_server_events", 1.0 if any(s == "server" for s in ctx.ev_source) else 0.0)
    out.add("client_event_share", sum(1 for s in ctx.ev_source if s == "client") / n_ev)
    out.add("server_event_share", sum(1 for s in ctx.ev_source if s == "server") / n_ev)
    out.add("unknown_source_share", sum(1 for s in ctx.ev_source if s not in ("client", "server")) / n_ev)
    out.add("target_present_share", float(ctx.ev_has_target.mean()) if ctx.n_events else 0.0)
    out.add("value_present_share", float(ctx.ev_has_value.mean()) if ctx.n_events else 0.0)
    out.add("point_value_share", len(ctx.points) / n_ev)
    out.add("dec_present_share", float(np.isfinite(ctx.a_dec).mean()) if ctx.n_actions else 0.0)
    out.add("tsla_present_share", float(np.isfinite(ctx.a_tsla).mean()) if ctx.n_actions else 0.0)
    out.add("occurred_at_share", float(np.isfinite(ctx.a_occ).mean()) if ctx.n_actions else 0.0)
    out.add("pot_present_share", float(np.isfinite(ctx.a_pot).mean()) if ctx.n_actions else 0.0)
    out.add("call_present_share", float(np.isfinite(ctx.a_call).mean()) if ctx.n_actions else 0.0)
    out.add("stack_present_share", float(np.isfinite(ctx.a_stack).mean()) if ctx.n_actions else 0.0)
    out.add("active_present_share", float(np.isfinite(ctx.a_active).mean()) if ctx.n_actions else 0.0)
    out.add("phase_present_share", float(np.isfinite(ctx.a_street).mean()) if ctx.n_actions else 0.0)
    out.add("board_present_share", float(np.isfinite(ctx.a_board).mean()) if ctx.n_actions else 0.0)
    out.add("hole_present_share", float(ctx.a_hole.mean()) if ctx.n_actions else 0.0)
    out.add("position_present_share", sum(1 for p in ctx.a_pos if p) / n_ac)
    out.add("actions_per_hand", ctx.n_actions / max(ctx.n_hands, 1))
    out.add("events_per_action", ctx.n_events / n_ac)
    out.add("log_duration_min", math.log10(1.0 + ctx.duration_ms / 60000.0))
    out.add("log_n_hands", math.log10(1.0 + ctx.n_hands))


# ---------------------------------------------------------------------------
# g02 summary block + consistency
# ---------------------------------------------------------------------------


def _g02_summary(ctx: _Ctx, out: _Out) -> None:
    s = ctx.summary
    ec = _num(s.get("event_count"), 0.0) or 0.0
    ac = _num(s.get("action_count"), 0.0) or 0.0
    dur = _num(s.get("duration_ms"), 0.0) or 0.0
    dc = _num(s.get("decision_count"), 0.0) or 0.0
    dmean = _num(s.get("decision_mean_ms"), 0.0) or 0.0
    dstd = _num(s.get("decision_std_ms"), 0.0) or 0.0

    out.add("event_count_log", math.log10(1.0 + max(ec, 0.0)))
    out.add("action_count_log", math.log10(1.0 + max(ac, 0.0)))
    out.add("duration_min_log", math.log10(1.0 + max(dur, 0.0) / 60000.0))
    out.add("decision_count_log", math.log10(1.0 + max(dc, 0.0)))
    out.add("decision_mean_log", math.log10(1.0 + max(dmean, 0.0)))
    out.add("decision_std_log", math.log10(1.0 + max(dstd, 0.0)))
    out.add("decision_cv", _rel(dstd, dmean))
    out.add("decision_std_zero", 1.0 if dstd <= _EPS else 0.0)
    out.add("decision_std_tiny", 1.0 if 0.0 < dstd < 25.0 else 0.0)
    out.add("decision_mean_sub300", 1.0 if 0.0 < dmean < 300.0 else 0.0)
    out.add("decision_mean_sub150", 1.0 if 0.0 < dmean < 150.0 else 0.0)
    out.add("events_per_min", ec / ctx.minutes)
    out.add("actions_per_min", ac / ctx.minutes)
    out.add("decisions_per_min", dc / ctx.minutes)
    out.add("events_per_action_sum", _safe_div(ec, max(ac, 1.0)))
    out.add("decision_per_action_sum", _safe_div(dc, max(ac, 1.0)))
    out.add("decision_time_share_of_session", _safe_div(dmean * dc, max(dur, 1.0)))
    # ---- summary vs observed consistency ---------------------------------
    obs_ec = float(ctx.n_events)
    obs_ac = float(ctx.n_actions)
    obs_dc = float(ctx.dec.size)
    obs_mean = float(ctx.dec.mean()) if ctx.dec.size else 0.0
    obs_std = float(ctx.dec.std()) if ctx.dec.size else 0.0
    out.add("ec_mismatch", _rel(abs(ec - obs_ec), max(ec, obs_ec)))
    out.add("ac_mismatch", _rel(abs(ac - obs_ac), max(ac, obs_ac)))
    out.add("dc_mismatch", _rel(abs(dc - obs_dc), max(dc, obs_dc)))
    out.add("dmean_ratio", _rel(obs_mean, dmean))
    out.add("dstd_ratio", _rel(obs_std, dstd))
    out.add("dmean_logdiff", math.log10(1.0 + max(obs_mean, 0.0)) - math.log10(1.0 + max(dmean, 0.0)))
    out.add("dstd_logdiff", math.log10(1.0 + max(obs_std, 0.0)) - math.log10(1.0 + max(dstd, 0.0)))
    span = float(ctx.ev_off.max() - ctx.ev_off.min()) if ctx.n_events >= 2 else 0.0
    out.add("span_vs_duration", _rel(span, dur))
    out.add("event_span_min_log", math.log10(1.0 + span / 60000.0))
    out.add("summary_all_zero", 1.0 if (ec + ac + dur + dc + dmean + dstd) <= 0.0 else 0.0)


# ---------------------------------------------------------------------------
# g03 inter-event interaction timing
# ---------------------------------------------------------------------------


def _g03_interact(ctx: _Ctx, out: _Out) -> None:
    dt = ctx.ev_dt
    ldt = _log10p(dt)
    _dist(out, "logdt", ldt)
    _dist(out, "dt", dt)
    for lo, hi, tag in (
        (0.0, 0.5, "zero"),
        (0.5, 16.0, "sub16"),
        (16.0, 50.0, "16_50"),
        (50.0, 150.0, "50_150"),
        (150.0, 500.0, "150_500"),
        (500.0, 2000.0, "500_2k"),
        (2000.0, 10000.0, "2k_10k"),
        (10000.0, 60000.0, "10k_60k"),
        (60000.0, float("inf"), "gt60k"),
    ):
        out.add(f"dt_share_{tag}", float(np.mean((dt >= lo) & (dt < hi))) if dt.size else 0.0)
    out.add("events_per_min", ctx.n_events / ctx.minutes)
    out.add("client_events_per_min", sum(1 for s in ctx.ev_source if s == "client") / ctx.minutes)
    out.add("events_per_hand", ctx.n_events / max(ctx.n_hands, 1))
    out.add("dt_entropy", _norm_entropy(Counter(np.floor(ldt * 5.0).astype(int).tolist()).values()) if dt.size else 0.0)
    out.add("dt_burstiness", _rel(float(dt.std()) - float(dt.mean()), float(dt.std()) + float(dt.mean())) if dt.size >= 3 else 0.0)
    out.add("dt_memory", _acf(dt, 1))
    out.add("dt_top3_gap_mass", _safe_div(float(np.sort(dt)[-3:].sum()), float(dt.sum())) if dt.size >= 6 else 0.0)
    out.add("dt_top_decile_mass", _safe_div(float(np.sort(dt)[-max(1, dt.size // 10):].sum()), float(dt.sum())) if dt.size >= 10 else 0.0)
    out.add("offset_monotonic", 1.0 if dt.size and bool(np.all(np.diff(ctx.ev_off) >= 0)) else 0.0)
    out.add("offset_start_ms_log", math.log10(1.0 + float(ctx.ev_off.min())) if ctx.n_events else 0.0)
    out.add("n_events_log", math.log10(1.0 + ctx.n_events))


# ---------------------------------------------------------------------------
# g04 poker decision timing
# ---------------------------------------------------------------------------


def _g04_decision(ctx: _Ctx, out: _Out) -> None:
    dec = ctx.dec
    tsla = ctx.tsla
    _dist(out, "logdec", _log10p(dec))
    _dist(out, "dec", dec)
    _dist(out, "logtsla", _log10p(tsla))
    for lo, hi, tag in (
        (0.0, 100.0, "sub100"),
        (100.0, 250.0, "100_250"),
        (250.0, 500.0, "250_500"),
        (500.0, 1000.0, "500_1k"),
        (1000.0, 3000.0, "1k_3k"),
        (3000.0, 8000.0, "3k_8k"),
        (8000.0, 20000.0, "8k_20k"),
        (20000.0, float("inf"), "gt20k"),
    ):
        out.add(f"dec_share_{tag}", float(np.mean((dec >= lo) & (dec < hi))) if dec.size else 0.0)
    out.add("dec_sub_reaction_share", float(np.mean(dec < 200.0)) if dec.size else 0.0)
    out.add("dec_timebank_share", float(np.mean(dec > 12000.0)) if dec.size else 0.0)
    out.add("dec_burstiness", _rel(float(dec.std()) - float(dec.mean()), float(dec.std()) + float(dec.mean())) if dec.size >= 3 else 0.0)
    out.add("dec_iqr_over_mad", _rel(float(np.subtract(*np.percentile(dec, [75, 25]))), float(np.abs(dec - np.median(dec)).mean())) if dec.size >= 4 else 0.0)
    out.add("dec_mode_share", (max(Counter(np.rint(dec).astype(int).tolist()).values()) / dec.size) if dec.size else 0.0)
    out.add("dec_unique_rate", (len(set(np.rint(dec).astype(int).tolist())) / dec.size) if dec.size else 0.0)
    out.add("dec_per_min", dec.size / ctx.minutes)
    out.add("tsla_over_dec", _rel(float(tsla.mean()) if tsla.size else 0.0, float(dec.mean()) if dec.size else 0.0))
    out.add("tsla_dec_corr", _spearman(ctx.a_tsla, ctx.a_dec))
    out.add("dec_sum_share_of_duration", _safe_div(float(dec.sum()), max(ctx.duration_ms, 1.0)))
    out.add("dec_lognormal_fit", _lognormal_fit(dec))
    out.add("dec_zero_share", float(np.mean(dec <= 0.0)) if dec.size else 0.0)


def _lognormal_fit(a: np.ndarray) -> float:
    """Crude lognormality score: corr of sorted log values with normal quantiles."""
    a = _finite(a)
    a = a[a > 0]
    if a.size < 12:
        return 0.0
    x = np.sort(np.log(a))
    p = (np.arange(1, x.size + 1) - 0.375) / (x.size + 0.25)
    # Acklam-free normal quantile via erfinv series (numpy has no ppf)
    q = np.sqrt(2.0) * _erfinv(2.0 * p - 1.0)
    return _pearson(x, q)


def _erfinv(y: np.ndarray) -> np.ndarray:
    """Vectorised inverse error function (Giles 2012 single-precision form)."""
    y = np.clip(np.asarray(y, float), -0.999999, 0.999999)
    w = -np.log((1.0 - y) * (1.0 + y))
    out = np.empty_like(w)
    m = w < 5.0
    ww = w[m] - 2.5
    p = 2.81022636e-08
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
        p = p * ww + c
    out[m] = p * y[m]
    ww = np.sqrt(w[~m]) - 3.0
    p = -0.000200214257
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
              -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
        p = p * ww + c
    out[~m] = p * y[~m]
    return out


# ---------------------------------------------------------------------------
# g05 timing regularity (quantisation / serial structure)
# ---------------------------------------------------------------------------


def _g05_regular(ctx: _Ctx, out: _Out) -> None:
    _grid_block(out, "dec", ctx.dec)
    _grid_block(out, "evdt", ctx.ev_dt)
    _grid_block(out, "tsla", ctx.tsla)
    _reg_block(out, "logdec", _log10p(ctx.dec))
    _reg_block(out, "logevdt", _log10p(ctx.ev_dt))
    # occurred_at derived structure
    occ = ctx.a_occ[np.isfinite(ctx.a_occ)]
    out.add("occ_present", 1.0 if occ.size >= 3 else 0.0)
    if occ.size >= 3:
        frac = np.abs(occ - np.floor(occ))
        ms = np.rint(frac * 1000.0).astype(int)
        out.add("occ_ms_zero_share", float(np.mean(ms == 0)))
        out.add("occ_ms_entropy", _norm_entropy(Counter((ms // 10).tolist()).values()))
        d = np.diff(np.sort(occ)) * 1000.0
        out.add("occ_dt_cv", _rel(float(d.std()), float(d.mean())))
        out.add("occ_span_vs_duration", _rel(float((occ.max() - occ.min()) * 1000.0), ctx.duration_ms))
        out.add("occ_dt_grid1000_share", float(np.mean(np.rint(d).astype(np.int64) % 1000 == 0)))
    else:
        out.add("occ_ms_zero_share", 0.0)
        out.add("occ_ms_entropy", 0.0)
        out.add("occ_dt_cv", 0.0)
        out.add("occ_span_vs_duration", 0.0)
        out.add("occ_dt_grid1000_share", 0.0)
    # cross-series: are decision times and event gaps the same process?
    out.add("dec_evdt_q50_ratio", _rel(float(np.median(ctx.dec)) if ctx.dec.size else 0.0,
                                       float(np.median(ctx.ev_dt)) if ctx.ev_dt.size else 0.0))
    out.add("dec_cv_minus_evdt_cv",
            (_rel(float(ctx.dec.std()), float(ctx.dec.mean())) if ctx.dec.size >= 3 else 0.0)
            - (_rel(float(ctx.ev_dt.std()), float(ctx.ev_dt.mean())) if ctx.ev_dt.size >= 3 else 0.0))


# ---------------------------------------------------------------------------
# g06 decision-time / complexity coupling  (the poker-specific edge)
# ---------------------------------------------------------------------------

_PROXIES = (
    "log_pot",
    "pot_odds",
    "call_over_pot",
    "active_players",
    "street",
    "act_idx",
    "log_spr",
    "bet_over_pot",
)


def _g06_coupling(ctx: _Ctx, out: _Out) -> None:
    y = ctx.log_dec_all  # nan where missing
    pot = ctx.a_pot
    call = ctx.a_call
    proxies = {
        "log_pot": _log10p(np.where(np.isfinite(pot), pot, np.nan)),
        "pot_odds": np.where(np.isfinite(pot) & np.isfinite(call), call / (np.abs(pot) + np.abs(call) + 1.0), np.nan),
        "call_over_pot": np.where(np.isfinite(pot) & np.isfinite(call), call / (np.abs(pot) + 1.0), np.nan),
        "active_players": ctx.a_active,
        "street": ctx.a_street,
        "act_idx": ctx.a_idx_norm,
        "log_spr": _log10p(np.where(np.isfinite(ctx.a_stack) & np.isfinite(pot), ctx.a_stack / (np.abs(pot) + 1.0), np.nan)),
        "bet_over_pot": np.where(np.isfinite(ctx.a_curbet) & np.isfinite(pot), ctx.a_curbet / (np.abs(pot) + 1.0), np.nan),
    }
    absmax = 0.0
    zmat: list[np.ndarray] = []
    for key in _PROXIES:
        x = proxies[key]
        rho = _spearman(x, y)
        out.add(f"rho_{key}", rho)
        absmax = max(absmax, abs(rho))
        out.add(f"hilo_{key}", _hi_lo_ratio(x, y))
        zmat.append(x)
    out.add("rho_absmax", absmax)
    out.add("coupling_flatness", 1.0 - absmax)
    out.add("coupling_r2", _multi_r2(zmat, y))

    # per-street normalised mean think time
    base = float(np.nanmean(y)) if np.isfinite(y).any() else 0.0
    street_means = []
    for i, name in enumerate(_STREETS):
        m = np.isfinite(y) & (ctx.a_street == float(i))
        v = float(y[m].mean() - base) if int(m.sum()) >= 3 else 0.0
        out.add(f"street_{name}_dlog", v)
        if int(m.sum()) >= 3:
            street_means.append(v)
    out.add("street_dlog_spread", float(np.std(street_means)) if len(street_means) >= 2 else 0.0)
    out.add("street_dlog_range", float(max(street_means) - min(street_means)) if len(street_means) >= 2 else 0.0)
    out.add("street_coverage", len(street_means) / 4.0)

    # per-action-type normalised mean think time (folds fast, raises slow)
    type_means = []
    types = np.array(ctx.a_type, dtype=object) if ctx.a_type else np.zeros(0, dtype=object)
    for name in _ACTION_TYPES:
        if types.size:
            m = np.isfinite(y) & (types == name)
        else:
            m = np.zeros(0, dtype=bool)
        v = float(y[m].mean() - base) if int(m.sum()) >= 3 else 0.0
        out.add(f"atype_{name}_dlog", v)
        if int(m.sum()) >= 3:
            type_means.append(v)
    out.add("atype_dlog_spread", float(np.std(type_means)) if len(type_means) >= 2 else 0.0)
    out.add("atype_dlog_range", float(max(type_means) - min(type_means)) if len(type_means) >= 2 else 0.0)
    out.add("atype_coverage", len(type_means) / float(len(_ACTION_TYPES)))
    # fold-vs-aggressive contrast: the single most human-legible coupling
    fold_m = np.isfinite(y) & (types == "fold") if types.size else np.zeros(0, dtype=bool)
    agg_m = np.isfinite(y) & np.isin(types, _AGGRESSIVE) if types.size else np.zeros(0, dtype=bool)
    out.add("fold_vs_agg_dlog",
            float(y[agg_m].mean() - y[fold_m].mean()) if int(fold_m.sum()) >= 3 and int(agg_m.sum()) >= 3 else 0.0)
    facing = np.isfinite(call) & (call > 0)
    out.add("facing_bet_dlog", _group_delta(y, facing))
    out.add("allin_dlog", _group_delta(y, ctx.a_allin > 0.5))
    out.add("first_action_dlog", _group_delta(y, ctx.a_idx_norm <= _EPS))
    # residual dispersion after conditioning on street+type: bots collapse to 0
    out.add("resid_std_ratio", _resid_std_ratio(y, ctx.a_street, types))


def _hi_lo_ratio(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 8:
        return 0.0
    xv, yv = x[m], y[m]
    med = float(np.median(xv))
    hi = yv[xv > med]
    lo = yv[xv <= med]
    if hi.size < 3 or lo.size < 3:
        return 0.0
    return float(hi.mean() - lo.mean())


def _group_delta(y: np.ndarray, mask: np.ndarray) -> float:
    if mask.size != y.size:
        return 0.0
    m = np.isfinite(y)
    a = y[m & mask]
    b = y[m & ~mask]
    if a.size < 3 or b.size < 3:
        return 0.0
    return float(a.mean() - b.mean())


def _multi_r2(cols: list[np.ndarray], y: np.ndarray) -> float:
    """R^2 of an OLS fit of y on the (standardised, mean-imputed) proxies."""
    m = np.isfinite(y)
    if int(m.sum()) < 12 or not cols:
        return 0.0
    yy = y[m]
    mat = []
    for c in cols:
        if c.size != y.size:
            continue
        v = c[m].astype(float)
        f = np.isfinite(v)
        if f.sum() < 4:
            continue
        fill = float(v[f].mean())
        v = np.where(f, v, fill)
        sd = v.std()
        if sd < _EPS:
            continue
        mat.append((v - v.mean()) / sd)
    if not mat:
        return 0.0
    x = np.column_stack(mat + [np.ones(yy.size)])
    if x.shape[0] <= x.shape[1] + 2:
        return 0.0
    try:
        beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    pred = x @ beta
    ss_res = float(((yy - pred) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    if ss_tot < _EPS:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))


def _resid_std_ratio(y: np.ndarray, street: np.ndarray, types: np.ndarray) -> float:
    m = np.isfinite(y)
    if int(m.sum()) < 12:
        return 0.0
    yy = y[m]
    keys = []
    for i in range(y.size):
        if not np.isfinite(y[i]):
            continue
        s = street[i] if np.isfinite(street[i]) else -1.0
        t = types[i] if types.size > i else ""
        keys.append((float(s), t))
    groups: dict[tuple, list[float]] = {}
    for k, v in zip(keys, yy):
        groups.setdefault(k, []).append(float(v))
    resid = []
    for vals in groups.values():
        if len(vals) < 2:
            continue
        mu = sum(vals) / len(vals)
        resid.extend(v - mu for v in vals)
    if len(resid) < 8:
        return 0.0
    total = float(yy.std())
    if total < _EPS:
        return 0.0
    return float(np.std(resid) / total)


# ---------------------------------------------------------------------------
# g07 motor signature
# ---------------------------------------------------------------------------


def _g07_motor(ctx: _Ctx, out: _Out) -> None:
    pts = ctx.points
    n_pts = len(pts)
    out.add("has_points", 1.0 if n_pts >= 3 else 0.0)
    out.add("points_per_min", n_pts / ctx.minutes)
    out.add("points_per_event", n_pts / max(ctx.n_events, 1))
    if n_pts >= 3:
        p = np.array(pts, dtype=float)
        d = np.diff(p, axis=0)
        step = np.hypot(d[:, 0], d[:, 1])
        path = float(step.sum())
        net = float(np.hypot(p[-1, 0] - p[0, 0], p[-1, 1] - p[0, 1]))
        t = np.array(ctx.point_dt, dtype=float)
        dtms = np.clip(np.diff(t), 1.0, None)
        speed = step / dtms
        ang = np.arctan2(d[:, 1], d[:, 0])
        dang = np.abs(np.diff(ang))
        dang = np.minimum(dang, 2 * math.pi - dang)
        frac_x = np.abs(p[:, 0] - np.rint(p[:, 0]))
        exact_rep = float(np.mean(step < _EPS))
        out.add("step_mean_log", math.log10(1.0 + float(step.mean())))
        out.add("step_cv", _rel(float(step.std()), float(step.mean())))
        out.add("step_q50_log", math.log10(1.0 + float(np.median(step))))
        out.add("straightness", _rel(net, path))
        out.add("speed_mean_log", math.log10(1.0 + float(speed.mean())))
        out.add("speed_cv", _rel(float(speed.std()), float(speed.mean())))
        out.add("speed_acf1", _acf(speed, 1))
        out.add("dir_change_mean", float(dang.mean()) if dang.size else 0.0)
        out.add("dir_change_std", float(dang.std()) if dang.size else 0.0)
        out.add("dir_reversal_rate", float(np.mean(dang > 2.0)) if dang.size else 0.0)
        out.add("exact_repeat_share", exact_rep)
        out.add("unique_point_rate", len({(round(a, 3), round(b, 3)) for a, b in pts}) / n_pts)
        out.add("top_point_share", max(Counter([(round(a, 3), round(b, 3)) for a, b in pts]).values()) / n_pts)
        out.add("fractional_coord_share", float(np.mean(frac_x > 1e-6)))
        out.add("x_std_log", math.log10(1.0 + float(p[:, 0].std())))
        out.add("y_std_log", math.log10(1.0 + float(p[:, 1].std())))
        out.add("xy_corr", _pearson(p[:, 0], p[:, 1]))
        out.add("jitter_index", _rel(float(np.median(step)), float(step.mean())))
        out.add("micro_move_share", float(np.mean((step > 0) & (step < 3.0))))
        out.add("teleport_share", float(np.mean(step > 500.0)))
        out.add("step_perm_ent", _perm_entropy(step, 3))
    else:
        for name in ("step_mean_log", "step_cv", "step_q50_log", "straightness", "speed_mean_log",
                     "speed_cv", "speed_acf1", "dir_change_mean", "dir_change_std", "dir_reversal_rate",
                     "exact_repeat_share", "unique_point_rate", "top_point_share", "fractional_coord_share",
                     "x_std_log", "y_std_log", "xy_corr", "jitter_index", "micro_move_share",
                     "teleport_share", "step_perm_ent"):
            out.add(name, 0.0)

    # click / target behaviour (works even when no coordinates are present)
    clicks = ctx.click_idx
    n_click = len(clicks)
    out.add("clicks_per_min", n_click / ctx.minutes)
    out.add("clicks_per_action", n_click / max(ctx.n_actions, 1))
    moves = sum(1 for et in ctx.ev_type if any(tok in et for tok in _MOVEY))
    out.add("move_to_click_ratio", _safe_div(float(moves), float(max(n_click, 1))))
    targets = [ctx.ev_target[i] for i in clicks if ctx.ev_target[i]]
    if targets:
        tc = Counter(targets)
        out.add("click_target_unique_rate", len(tc) / len(targets))
        out.add("click_target_top_share", max(tc.values()) / len(targets))
        out.add("click_target_entropy", _norm_entropy(tc.values()))
    else:
        out.add("click_target_unique_rate", 0.0)
        out.add("click_target_top_share", 0.0)
        out.add("click_target_entropy", 0.0)
    if n_click >= 3:
        coff = ctx.ev_off[np.array(clicks, dtype=int)]
        cdt = np.diff(coff)
        same = np.array(
            [1.0 if ctx.ev_target[clicks[i]] and ctx.ev_target[clicks[i]] == ctx.ev_target[clicks[i - 1]] else 0.0
             for i in range(1, n_click)],
            dtype=float,
        )
        out.add("double_click_rate", float(np.mean((cdt < 500.0) & (same > 0.5))))
        out.add("rapid_click_rate", float(np.mean(cdt < 250.0)))
        out.add("same_target_repeat_rate", float(same.mean()))
        out.add("click_dt_cv", _rel(float(cdt.std()), float(cdt.mean())))
        out.add("click_dt_q50_log", math.log10(1.0 + float(np.median(cdt))))
        out.add("click_dt_grid_share", float(np.mean(np.rint(cdt).astype(np.int64) % 100 == 0)))
    else:
        for name in ("double_click_rate", "rapid_click_rate", "same_target_repeat_rate",
                     "click_dt_cv", "click_dt_q50_log", "click_dt_grid_share"):
            out.add(name, 0.0)
    # hesitation: a click followed by another click on a DIFFERENT target
    # before the committed action (mind-changing at the bet slider)
    switches = 0
    for i in range(1, n_click):
        a, b = ctx.ev_target[clicks[i - 1]], ctx.ev_target[clicks[i]]
        if a and b and a != b and (ctx.ev_off[clicks[i]] - ctx.ev_off[clicks[i - 1]]) < 1500.0:
            switches += 1
    out.add("hesitation_switch_rate", switches / max(n_click - 1, 1))


# ---------------------------------------------------------------------------
# g08 attention / fatigue
# ---------------------------------------------------------------------------


def _g08_attention(ctx: _Ctx, out: _Out) -> None:
    n_ev = max(ctx.n_events, 1)
    fam_counts = {k: 0 for k in _ATTENTION_KEYS}
    for et, tg in zip(ctx.ev_type, ctx.ev_target):
        blob = et + "|" + tg
        for fam, toks in _ATTENTION_FAMILIES.items():
            if any(tok in blob for tok in toks):
                fam_counts[fam] += 1
    for fam in _ATTENTION_KEYS:
        out.add(f"share_{fam}", fam_counts[fam] / n_ev)
        out.add(f"permin_{fam}", fam_counts[fam] / ctx.minutes)
    out.add("attention_family_coverage", sum(1 for v in fam_counts.values() if v > 0) / float(len(_ATTENTION_KEYS)))
    out.add("blur_focus_ratio", _rel(float(fam_counts["blur"]), float(fam_counts["focus"])))
    out.add("evtype_unique_rate", len(set(ctx.ev_type)) / n_ev)
    out.add("evtype_entropy", _norm_entropy(Counter(ctx.ev_type).values()))
    out.add("evtype_top_share", (max(Counter(ctx.ev_type).values()) / n_ev) if ctx.n_events else 0.0)
    out.add("target_unique_rate", len({t for t in ctx.ev_target if t}) / n_ev)
    out.add("target_entropy", _norm_entropy(Counter([t for t in ctx.ev_target if t]).values()))

    dt = ctx.ev_dt
    dur = max(ctx.duration_ms, 1.0)
    for thr, tag in ((2000.0, "2s"), (5000.0, "5s"), (30000.0, "30s"), (120000.0, "120s")):
        gaps = dt[dt > thr]
        out.add(f"gap_{tag}_permin", gaps.size / ctx.minutes)
        out.add(f"gap_{tag}_duration_share", float(gaps.sum()) / dur)
    out.add("longest_gap_share", (float(dt.max()) / dur) if dt.size else 0.0)
    out.add("idle_mass", float(dt[dt > 5000.0].sum()) / dur if dt.size else 0.0)
    out.add("active_density", float(np.mean(dt < 2000.0)) if dt.size else 0.0)

    # fatigue / drift over the session (action-level)
    y = ctx.log_dec_all
    m = np.isfinite(y)
    if int(m.sum()) >= 12:
        yy = y[m]
        prog = np.linspace(0.0, 1.0, yy.size)
        out.add("fatigue_slope", _ols_slope(prog, yy))
        out.add("fatigue_rho", _spearman(prog, yy))
        third = max(yy.size // 3, 1)
        f, l = yy[:third], yy[-third:]
        out.add("third_first_last_dlog", float(l.mean() - f.mean()))
        out.add("third_std_ratio", _rel(float(l.std()), float(f.std())))
        out.add("half_dlog", float(yy[yy.size // 2:].mean() - yy[: yy.size // 2].mean()))
    else:
        for name in ("fatigue_slope", "fatigue_rho", "third_first_last_dlog", "third_std_ratio", "half_dlog"):
            out.add(name, 0.0)
    # event-rate drift across session thirds
    if ctx.n_events >= 12:
        off = ctx.ev_off
        lo, hi = float(off.min()), float(off.max())
        span = max(hi - lo, 1.0)
        rel = (off - lo) / span
        shares = [float(np.mean((rel >= a) & (rel < b))) for a, b in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0001))]
        out.add("evrate_third1", shares[0])
        out.add("evrate_third2", shares[1])
        out.add("evrate_third3", shares[2])
        out.add("evrate_drift", shares[2] - shares[0])
        out.add("evrate_third_std", float(np.std(shares)))
    else:
        for name in ("evrate_third1", "evrate_third2", "evrate_third3", "evrate_drift", "evrate_third_std"):
            out.add(name, 0.0)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 4:
        return 0.0
    vx = float(((x - x.mean()) ** 2).sum())
    if vx < _EPS:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / vx)


# ---------------------------------------------------------------------------
# g09 sequence regularity
# ---------------------------------------------------------------------------


def _time_bucket(ms: float) -> int:
    if not math.isfinite(ms):
        return -1
    for i, b in enumerate(_TIME_BUCKETS):
        if ms < b:
            return i
    return len(_TIME_BUCKETS)


def _g09_sequence(ctx: _Ctx, out: _Out) -> None:
    # (action, timing bucket) tokens -- the joint policy+tempo stream
    tokens_at = []
    tokens_a = []
    tokens_ast = []
    for i, t in enumerate(ctx.a_type):
        tb = _time_bucket(float(ctx.a_dec[i]) if i < ctx.a_dec.size else float("nan"))
        st = int(ctx.a_street[i]) if i < ctx.a_street.size and np.isfinite(ctx.a_street[i]) else -1
        tokens_at.append(f"{t}:{tb}")
        tokens_a.append(t)
        tokens_ast.append(f"{t}:{st}")
    _seq_block(out, "act_time", tokens_at)
    _seq_block(out, "act", tokens_a)
    _seq_block(out, "act_street", tokens_ast)
    _seq_block(out, "evtype", ctx.ev_type)
    ev_tok = []
    for i in range(ctx.n_events):
        gap = _time_bucket(float(ctx.ev_dt[i - 1]) if 0 < i <= ctx.ev_dt.size else float("nan"))
        ev_tok.append(f"{ctx.ev_type[i]}:{gap}")
    _seq_block(out, "ev_time", ev_tok)

    # repeated-line detection across hands
    sigs = ["|".join(str(a.get("action_type") or "") for a in acts) for acts in ctx.hand_actions]
    sigs_ns = [
        "|".join(f"{a.get('action_type')}:{a.get('phase')}" for a in acts) for acts in ctx.hand_actions
    ]
    n_h = max(len(sigs), 1)
    c1 = Counter(sigs)
    c2 = Counter(sigs_ns)
    out.add("hand_sig_unique_rate", len(c1) / n_h)
    out.add("hand_sig_top_share", (max(c1.values()) / n_h) if sigs else 0.0)
    out.add("hand_sig_dup_share", sum(v for v in c1.values() if v > 1) / n_h if sigs else 0.0)
    out.add("hand_sig_entropy", _norm_entropy(c1.values()))
    out.add("hand_sigp_unique_rate", len(c2) / n_h)
    out.add("hand_sigp_top_share", (max(c2.values()) / n_h) if sigs_ns else 0.0)
    out.add("hand_sigp_dup_share", sum(v for v in c2.values() if v > 1) / n_h if sigs_ns else 0.0)
    out.add("hand_sigp_entropy", _norm_entropy(c2.values()))
    # action-type mix (policy fingerprint, order invariant)
    tc = Counter(ctx.a_type)
    n_a = max(ctx.n_actions, 1)
    for name in _ACTION_TYPES:
        out.add(f"mix_{name}", tc.get(name, 0) / n_a)
    out.add("mix_entropy", _norm_entropy([tc.get(k, 0) for k in _ACTION_TYPES]))
    out.add("aggression_rate", sum(tc.get(k, 0) for k in _AGGRESSIVE) / n_a)
    out.add("passive_rate", sum(tc.get(k, 0) for k in _PASSIVE) / n_a)
    out.add("evtype_action_evtype_unique", len(set(ctx.a_evtype)) / n_a)


# ---------------------------------------------------------------------------
# g10 cross-hand drift / stationarity
# ---------------------------------------------------------------------------


def _g10_drift(ctx: _Ctx, out: _Out) -> None:
    n_h = ctx.n_hands
    per_mean: list[float] = []
    per_std: list[float] = []
    per_n: list[float] = []
    per_agg: list[float] = []
    per_min: list[float] = []
    for hi, acts in enumerate(ctx.hand_actions):
        m = ctx.a_hand == hi
        y = ctx.log_dec_all[m] if ctx.log_dec_all.size else np.zeros(0)
        y = y[np.isfinite(y)]
        if y.size >= 1:
            per_mean.append(float(y.mean()))
            per_std.append(float(y.std()))
            per_min.append(float(y.min()))
        types = [str(a.get("action_type") or "").lower() for a in acts]
        per_n.append(float(len(acts)))
        per_agg.append(sum(1 for t in types if t in _AGGRESSIVE) / max(len(types), 1))

    series = {
        "mean": np.array(per_mean, dtype=float),
        "agg": np.array(per_agg, dtype=float),
        "nact": np.array(per_n, dtype=float),
    }
    for key, arr in series.items():
        out.add(f"{key}_present", 1.0 if arr.size >= 4 else 0.0)
        out.add(f"{key}_cv", _rel(float(arr.std()), float(arr.mean())) if arr.size >= 4 else 0.0)
        out.add(f"{key}_acf1", _acf(arr, 1))
        out.add(f"{key}_trend_rho", _spearman(np.arange(arr.size, dtype=float), arr) if arr.size >= 4 else 0.0)
        out.add(f"{key}_slope", _ols_slope(np.arange(arr.size, dtype=float), arr) if arr.size >= 4 else 0.0)
        out.add(f"{key}_half_diff", _half_diff(arr))
        out.add(f"{key}_cusum", _cusum_max(arr))
        out.add(f"{key}_ks", _ks_halves(arr))
        out.add(f"{key}_runs_z", _runs_z(arr))
        out.add(f"{key}_range_rel", _rel(float(arr.max() - arr.min()), float(np.abs(arr).mean())) if arr.size >= 4 else 0.0)

    ps = np.array(per_std, dtype=float)
    pm = np.array(per_mean, dtype=float)
    all_y = ctx.log_dec
    out.add("within_std_mean", float(ps.mean()) if ps.size else 0.0)
    out.add("within_std_cv", _rel(float(ps.std()), float(ps.mean())) if ps.size >= 3 else 0.0)
    out.add("between_over_within", _rel(float(pm.std()), float(ps.mean())) if pm.size >= 3 and ps.size else 0.0)
    out.add("icc_like", _rel(float(pm.var()), float(pm.var()) + float(ps.mean() ** 2)) if pm.size >= 3 and ps.size else 0.0)
    out.add("within_over_total", _rel(float(ps.mean()), float(all_y.std())) if ps.size and all_y.size >= 3 else 0.0)
    out.add("hand_min_dlog_cv", _rel(float(np.std(per_min)), float(np.mean(per_min))) if len(per_min) >= 3 else 0.0)
    out.add("hands_with_timing_share", len(per_mean) / max(n_h, 1))
    out.add("n_hands_log", math.log10(1.0 + n_h))
    # session-level "one distribution?" test: max deviation of per-hand mean
    if pm.size >= 4 and pm.std() > _EPS:
        z = (pm - pm.mean()) / pm.std()
        out.add("hand_mean_absmax_z", float(np.abs(z).max()))
        out.add("hand_mean_outlier_share", float(np.mean(np.abs(z) > 2.0)))
    else:
        out.add("hand_mean_absmax_z", 0.0)
        out.add("hand_mean_outlier_share", 0.0)


def _half_diff(a: np.ndarray) -> float:
    a = _finite(a)
    if a.size < 6:
        return 0.0
    h = a.size // 2
    x, y = a[:h], a[h:]
    pooled = math.sqrt((x.var() + y.var()) / 2.0)
    if pooled < _EPS:
        return 0.0
    return float((y.mean() - x.mean()) / pooled)


def _cusum_max(a: np.ndarray) -> float:
    a = _finite(a)
    if a.size < 6:
        return 0.0
    sd = float(a.std())
    if sd < _EPS:
        return 0.0
    z = (a - a.mean()) / sd
    c = np.cumsum(z)
    return float(np.abs(c).max() / math.sqrt(a.size))


def _ks_halves(a: np.ndarray) -> float:
    a = _finite(a)
    if a.size < 10:
        return 0.0
    h = a.size // 2
    x = np.sort(a[:h])
    y = np.sort(a[h:])
    grid = np.union1d(x, y)
    cx = np.searchsorted(x, grid, side="right") / x.size
    cy = np.searchsorted(y, grid, side="right") / y.size
    return float(np.abs(cx - cy).max())


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

GROUP_ORDER = (
    "g01_presence",
    "g02_summary",
    "g03_interact",
    "g04_decision",
    "g05_regular",
    "g06_coupling",
    "g07_motor",
    "g08_attention",
    "g09_sequence",
    "g10_drift",
)

_GROUP_FUNCS = {
    "g01_presence": _g01_presence,
    "g02_summary": _g02_summary,
    "g03_interact": _g03_interact,
    "g04_decision": _g04_decision,
    "g05_regular": _g05_regular,
    "g06_coupling": _g06_coupling,
    "g07_motor": _g07_motor,
    "g08_attention": _g08_attention,
    "g09_sequence": _g09_sequence,
    "g10_drift": _g10_drift,
}

FEATURE_GROUPS: dict[str, list[str]] = {}
FEATURE_GROUP_INDICES: dict[str, list[int]] = {}
FEATURE_NAMES: list[str] = []
N_FEATURES = 0


def _run(session: Any) -> tuple[list[str], list[float]]:
    try:
        ctx = _Ctx(session)
    except Exception:  # pragma: no cover - context building is fully guarded
        ctx = _Ctx({})
    names: list[str] = []
    values: list[float] = []
    for group in GROUP_ORDER:
        fn = _GROUP_FUNCS[group]
        out = _Out(group)
        failed = False
        try:
            fn(ctx, out)
        except Exception:
            failed = True
        expected = FEATURE_GROUPS.get(group)
        if failed or (expected is not None and len(out.values) != len(expected)):
            if expected is None:
                raise RuntimeError(f"bootstrap failure in {group}")  # import-time only
            names.extend(expected)
            values.extend([0.0] * len(expected))
        else:
            names.extend(out.names)
            values.extend(out.values)
    return names, values


def extract_session_features(session: Any) -> tuple[list[str], list[float]]:
    """Return ``(FEATURE_NAMES, values)`` for one v3 subject session.

    Never raises. Missing/None/malformed telemetry yields zeros plus the
    presence flags in ``g01_presence``.
    """
    try:
        names, values = _run(session)
    except Exception:
        return list(FEATURE_NAMES), [0.0] * N_FEATURES
    if len(values) != N_FEATURES:  # defensive: should be impossible
        values = (values + [0.0] * N_FEATURES)[:N_FEATURES]
        names = list(FEATURE_NAMES)
    return names, values


def extract_batch(sessions) -> tuple[list[str], np.ndarray]:
    """Vectorise a list of sessions into an (n, N_FEATURES) float array."""
    rows = []
    for session in sessions or []:
        rows.append(extract_session_features(session)[1])
    if not rows:
        return list(FEATURE_NAMES), np.zeros((0, N_FEATURES), dtype=float)
    return list(FEATURE_NAMES), np.asarray(rows, dtype=float)


def _bootstrap() -> None:
    """Derive the fixed feature schema by running every group on ``{}``."""
    global FEATURE_NAMES, N_FEATURES
    ctx = _Ctx({})
    for group in GROUP_ORDER:
        out = _Out(group)
        _GROUP_FUNCS[group](ctx, out)
        FEATURE_GROUPS[group] = list(out.names)
    names: list[str] = []
    for group in GROUP_ORDER:
        start = len(names)
        names.extend(FEATURE_GROUPS[group])
        FEATURE_GROUP_INDICES[group] = list(range(start, len(names)))
    FEATURE_NAMES = names
    N_FEATURES = len(names)
    dupes = [n for n, c in Counter(names).items() if c > 1]
    if dupes:
        raise RuntimeError(f"duplicate feature names: {dupes[:5]}")


_bootstrap()
