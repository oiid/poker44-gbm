"""Chunk feature extraction v2 for Poker44 bot detection (STAGING).

Strict superset of the deployed v1 feature set (neurons/chunk_features.py,
111 features): the first N_FEATURES_V1 values are produced by the deployed
``_build`` and are bit-identical to v1 output.  Appended: cross-hand
SIGNATURE features (bots replay canonical action lines across hands; the
benchmark's ``pattern_hardened_v2`` equalizes per-label action-type
histograms, so per-chunk distinctness/sequencing is the surviving label
signal) plus per-hand-scalar dispersion features v1 lacks.

All new features are size-invariant (shares / rates / normalized entropies,
never raw counts) so 30-40-hand benchmark sub-chunks and 80-105-hand live
chunks live on the same scale.  Never raises on malformed input: any failure
returns a zero vector.

Deploy story: this file is imported alongside (not instead of) the frozen
v1 module; at the epoch boundary model_miner.py switches its import to
``chunk_features_v2``.  Stdlib-only, like v1.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    from neurons.chunk_features import (  # noqa: F401
        FEATURE_NAMES as V1_FEATURE_NAMES,
        _build as _build_v1,
        _mean,
        _safe_float,
        _std,
    )
except ImportError:  # bare-script use without PYTHONPATH=repo root
    sys.path.insert(0, "/root/bittensor/Poker44-subnet")
    from neurons.chunk_features import (  # noqa: F401
        FEATURE_NAMES as V1_FEATURE_NAMES,
        _build as _build_v1,
        _mean,
        _safe_float,
        _std,
    )

N_FEATURES_V1 = len(V1_FEATURE_NAMES)

_STREET_CODE = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}
_STREET_INDEX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_ACTION_CODE = {"fold": "f", "check": "k", "call": "c", "bet": "b", "raise": "r"}


def _amt_bucket(amt: float) -> str:
    """Coarse log2 bucket of a bb amount; 'z' for zero/invalid.

    Buckets are ordinal codes, used only as sequence-token components, so the
    absolute bucket boundaries only need to be stable, not domain-matched
    (signature features measure repetition, not magnitude).
    """
    if amt <= 0:
        return "z"
    b = int(math.floor(math.log2(amt)))
    b = max(-2, min(7, b))
    return str(b + 2)  # "0".."9"


def _entropy_norm(counts) -> float:
    """Shannon entropy of a count distribution, normalized to [0, 1]."""
    total = sum(counts)
    k = len(counts)
    if total <= 0 or k < 2:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h / math.log(k) if k > 1 else 0.0


def _seq_stats(seqs, prefix, names, out, second=False):
    """Distribution + run statistics of per-hand canonical sequences.

    seqs: list of hashable per-hand signatures in chunk hand order
    (None entries = unparseable hands, already filtered by caller).
    Emits shares/rates only.
    """
    cols = [
        f"{prefix}_top_share", f"{prefix}_unique_share",
        f"{prefix}_entropy", f"{prefix}_dup_pair_rate",
        f"{prefix}_max_run_share", f"{prefix}_runs_ratio",
    ]
    if second:
        cols.append(f"{prefix}_second_share")
    names.extend(cols)
    if out is None:
        return
    n = len(seqs)
    if n == 0:
        out.extend([0.0] * len(cols))
        return
    counts = {}
    for s in seqs:
        counts[s] = counts.get(s, 0) + 1
    cvals = sorted(counts.values(), reverse=True)
    top_share = cvals[0] / n
    unique_share = len(cvals) / n
    ent = _entropy_norm(cvals) if len(cvals) > 1 else 0.0
    # probability that two distinct hands in the chunk share the signature
    dup = (sum(c * (c - 1) for c in cvals) / (n * (n - 1))) if n > 1 else 0.0
    # consecutive-run structure (bots often replay lines back-to-back)
    max_run, cur_run, n_runs = 1, 1, 1
    for a, b in zip(seqs, seqs[1:]):
        if a == b:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            n_runs += 1
            cur_run = 1
    vals = [top_share, unique_share, ent, dup, max_run / n, n_runs / n]
    if second:
        vals.append((cvals[1] / n) if len(cvals) > 1 else 0.0)
    out.extend(vals)


def _hand_view(hand):
    """Parse one hand into token sequences and per-hand scalars.

    Returns None when the hand yields no usable actions.
    """
    if not isinstance(hand, dict):
        return None
    meta = hand.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    hero_seat = meta.get("hero_seat")
    actions = hand.get("actions") or []
    if not isinstance(actions, list):
        return None

    sa, saa, role, at = [], [], [], []
    actors = []
    type_counts = {}
    max_street = 0
    hero_positions = []
    hero_first = None
    n_valid = 0
    for a in actions:
        if not isinstance(a, dict):
            continue
        atype = a.get("action_type")
        street = a.get("street")
        ac = _ACTION_CODE.get(atype)
        sc = _STREET_CODE.get(street)
        if ac is None:
            continue
        n_valid += 1
        scode = sc if sc is not None else "x"
        amt = _safe_float(a.get("normalized_amount_bb"))
        sa.append(scode + ac)
        saa.append(scode + ac + _amt_bucket(amt))
        at.append(ac)
        is_hero = hero_seat is not None and a.get("actor_seat") == hero_seat
        role.append(("H" if is_hero else "O") + ac)
        actors.append(a.get("actor_seat"))
        type_counts[ac] = type_counts.get(ac, 0) + 1
        if street in _STREET_INDEX:
            max_street = max(max_street, _STREET_INDEX[street])
        if is_hero:
            hero_positions.append(n_valid - 1)
            if hero_first is None:
                hero_first = scode + ac
    if n_valid == 0:
        return None

    switches = sum(1 for x, y in zip(actors, actors[1:]) if x != y)
    switch_rate = switches / (n_valid - 1) if n_valid > 1 else 0.0
    return {
        "sa": ",".join(sa),
        "saa": ",".join(saa),
        "role": ",".join(role),
        "at": ",".join(at),
        "sa_set": frozenset(sa),
        "type_entropy": _entropy_norm(list(type_counts.values())),
        "switch_rate": switch_rate,
        "max_street": max_street / 3.0,
        "hero_pos": (_mean(hero_positions) / max(n_valid - 1, 1))
        if hero_positions else None,
        "hero_first": hero_first,
    }


def _build_sig(chunk, names, out):
    """Register (and optionally compute) the v2 signature feature block."""
    hands = chunk if isinstance(chunk, (list, tuple)) else []
    views = []
    if out is not None:
        for h in hands:
            try:
                v = _hand_view(h)
            except Exception:  # noqa: BLE001 - malformed hand -> skip
                v = None
            if v is not None:
                views.append(v)

    # ---- sequence-signature distributions at 4 canonicalizations ----------
    _seq_stats([v["sa"] for v in views], "sig_sa", names, out, second=True)
    _seq_stats([v["saa"] for v in views], "sig_saa", names, out, second=True)
    _seq_stats([v["role"] for v in views], "sig_role", names, out)
    _seq_stats([v["at"] for v in views], "sig_at", names, out)

    def emit(name, value):
        names.append(name)
        if out is not None:
            out.append(_safe_float(value))

    n = len(views)
    # ---- cross-hand token-set overlap (order-free duplication tell) -------
    names.extend(["sig_jaccard_mean", "sig_jaccard_std"])
    if out is not None:
        jacc = []
        if n >= 2:
            sets = [v["sa_set"] for v in views]
            for i in range(n - 1):
                si = sets[i]
                for j in range(i + 1, n):
                    sj = sets[j]
                    union = len(si | sj)
                    jacc.append(len(si & sj) / union if union else 0.0)
        out.extend([_mean(jacc), _std(jacc)])

    # ---- per-hand scalar dispersions (v1 has none of these) ----------------
    te = [v["type_entropy"] for v in views]
    emit("hand_type_entropy_mean", _mean(te))
    emit("hand_type_entropy_std", _std(te))
    sw = [v["switch_rate"] for v in views]
    emit("hand_actor_switch_rate_mean", _mean(sw))
    emit("hand_actor_switch_rate_std", _std(sw))
    ms = [v["max_street"] for v in views]
    emit("hand_max_street_mean", _mean(ms))
    emit("hand_max_street_std", _std(ms))
    hp = [v["hero_pos"] for v in views if v["hero_pos"] is not None]
    emit("hand_hero_pos_mean", _mean(hp))
    emit("hand_hero_pos_std", _std(hp))
    # distribution entropy of hero's first visible action token
    names.append("hero_first_action_entropy")
    if out is not None:
        hf = {}
        for v in views:
            if v["hero_first"] is not None:
                hf[v["hero_first"]] = hf.get(v["hero_first"], 0) + 1
        out.append(_entropy_norm(list(hf.values())) if len(hf) > 1 else 0.0)


def _feature_names():
    names = list(V1_FEATURE_NAMES)
    _build_sig([], names, None)
    return names


FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)
SIG_FEATURE_NAMES = FEATURE_NAMES[N_FEATURES_V1:]


def extract_features(chunk):
    """Fixed-length v2 feature vector: v1 (bit-identical) + signature block.

    Never raises: on any failure returns a zero vector of length N_FEATURES.
    """
    try:
        names, out = [], []
        _build_v1(chunk, names, out)
        _build_sig(chunk, names, out)
        if len(out) != N_FEATURES:
            return [0.0] * N_FEATURES
        return [_safe_float(v) for v in out]
    except Exception:  # noqa: BLE001
        return [0.0] * N_FEATURES


if __name__ == "__main__":
    print(f"{N_FEATURES} features ({N_FEATURES_V1} v1 + "
          f"{N_FEATURES - N_FEATURES_V1} new)")
    for n in FEATURE_NAMES[N_FEATURES_V1:]:
        print(" ", n)
