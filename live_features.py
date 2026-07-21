"""
live_features.py -- live-robust ACTION N-GRAM feature bank for SN126 Poker44.

WHY THIS IS NOT A RAW N-GRAM BANK
---------------------------------
The first cut of this bank used raw n-gram SHARES.  Measured against the live
capture it was 48% live-collapsed (median KS 0.567), because the live population
simply plays a different action mix than the sanitized benchmark:

    action     bench share   live share
    fold          0.50          0.35
    check         0.17          0.30
    call          0.15          0.28

Every raw share therefore shifts whether or not the hero is a bot.  Measured
comparison of five normalizations over the same 346 n-grams (see
experiment_norms.py, output in norm_experiment.log):

    norm    medKS   KS>=0.6   %collapsed   n(|AUC-0.5|>=0.05)
    share   0.567     162       48.3%              18
    lift    0.549     117       50.0%              17
    clr     0.474     126       37.6%              83
    rank    0.406     119       36.1%              44
    cond    0.378      55       22.4%              28     <-- adopted

So the bank ships CONDITIONAL TRANSITION PROBABILITIES P(next | prefix) for
n>=2 and WITHIN-BLOCK RANKS for n=1.  Conditioning divides out the first-order
action mix, which is the part of the payload that shifts; what is left is the
sequential dependence structure, which is what a bot policy actually distorts.

DESIGN CONTRACT
  * NO raw magnitudes.  Stacks, pots and bb amounts never reach a feature.  Bet
    size enters only as a WITHIN-CHUNK tercile bucket, invariant to any monotone
    marginal shift by construction.
  * NO vocabulary-size / diversity leakage.  The quadrant-(ii) mirage columns
    (`*_unique_rate`, `*_singleton_share`) exploded live because live payloads
    mix 6/7/8/9-max tables and are 2-3x longer per chunk.  Alphabets here are
    FIXED and table-size independent (positions are relative, capped at 3
    classes), and no feature counts distinct symbols.
  * SIZE INVARIANCE: every feature is a conditional probability, a rank, or a
    per-hand mean.  A 40-hand chunk and a 100-hand chunk are comparable.
  * HAND-ORDER INVARIANCE: n-grams are extracted strictly WITHIN a hand and then
    pooled by summation.
  * TRUNCATION ROBUSTNESS: live hands expose 5-8 actions, benchmark hands 1-19.
    No end-of-hand sentinel is emitted; that would encode truncation.
  * EMPIRICAL DROP LIST: features whose live distribution still collapses
    (KS >= 0.60 or live variance ratio < 0.05) are frozen out in
    live_drop_list.json, which is fitted on OLDER captures only and validated on
    the newest one.

Public API
  FEATURE_NAMES            fixed-arity list[str]
  extract_features(chunk)  list[float], never raises, always finite
  extract_matrix(chunks)   np.ndarray
  within_batch_rank(X)     pd-coast within-request column-rank transform
  GROUPS                   dict[str, list[int]] of feature-index groups

stdlib + numpy only.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "GROUPS",
    "extract_features",
    "extract_matrix",
    "within_batch_rank",
    "tokenize_hand",
    "VOCAB_PATH",
    "DROP_PATH",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(_HERE, "ngram_vocab.json")
DROP_PATH = os.path.join(_HERE, "live_drop_list.json")

# ---------------------------------------------------------------------------
# alphabets (fixed, table-size independent)
# ---------------------------------------------------------------------------

_ACT_CODE = {"fold": "f", "check": "k", "call": "c", "bet": "b", "raise": "r"}
_ST_CODE = {"preflop": "P", "flop": "F", "turn": "T", "river": "R"}
_SIZED = ("call", "bet", "raise")
_POS = ("E", "M", "L")
_SIZE_BUCKETS = ("L", "M", "H")

_BLOCKS = (
    ("A", 1), ("A", 2), ("A", 3),      # action_type
    ("SA", 1), ("SA", 2), ("SA", 3),   # street + action_type
    ("AZ", 1), ("AZ", 2), ("AZ", 3),   # action_type + within-chunk size bucket
    ("RA", 1), ("RA", 2),              # hero-flag + relative position + action
    ("HR", 1), ("HR", 2),              # hero-only stream: street + action
)

_SMOOTH = 1e-3
# Shrinkage of sparse conditionals toward the chunk marginal, expressed as a
# FRACTION of the block's mean prefix support rather than as an absolute
# pseudo-count.  An absolute pseudo-count would shrink a 40-hand benchmark chunk
# harder than a 90-hand live chunk and so manufacture exactly the kind of
# size-driven domain shift this bank exists to avoid; scaling it with the sample
# keeps the transform exactly invariant to replicating the chunk.
# Overridable only so the build/validate scripts can A/B it.
_COND_ALPHA_REL = float(os.environ.get("P44_COND_ALPHA_REL", "0.3"))

# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------


def _f(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _hand_actions(hand):
    """Normalized action records of one hand, [] if malformed."""
    if not isinstance(hand, dict):
        return []
    meta = hand.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    hero = meta.get("hero_seat")
    raw = hand.get("actions")
    if not isinstance(raw, list):
        return []
    out = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        act = a.get("action_type")
        if act not in _ACT_CODE:
            continue
        st = a.get("street")
        st = st if st in _ST_CODE else "preflop"
        amt = _f(a.get("amount"), 0.0)
        pot = _f(a.get("pot_before"), 0.0)
        ratio = (amt / pot) if (pot > 1e-9 and amt > 1e-9) else None
        seat = a.get("actor_seat")
        out.append({
            "street": st, "act": act, "seat": seat,
            "hero": (seat is not None and hero is not None and seat == hero),
            "ratio": ratio,
        })
    return out


def _position_classes(acts):
    """seat -> relative position class {E,M,L}.

    Rank of the seat's FIRST action in the hand, normalized by the number of
    actors actually seen.  Normalizing by observed actors (not by max_seats) is
    what makes this survive the live 6/7/8/9-max mixture.
    """
    order = []
    for r in acts:
        if r["seat"] not in order:
            order.append(r["seat"])
    n = len(order)
    if n <= 0:
        return {}
    cls = {}
    for i, s in enumerate(order):
        q = (i + 0.5) / n
        cls[s] = _POS[0] if q < 1.0 / 3 else (_POS[1] if q < 2.0 / 3 else _POS[2])
    return cls


def _size_bucket(ratio, cuts):
    if ratio is None or cuts is None:
        return "x"
    if ratio <= cuts[0]:
        return _SIZE_BUCKETS[0]
    if ratio <= cuts[1]:
        return _SIZE_BUCKETS[1]
    return _SIZE_BUCKETS[2]


def tokenize_hand(acts, gran, cuts=None):
    """Token sequence of one hand at granularity gran in {A,SA,AZ,RA,HR}."""
    if not acts:
        return []
    if gran == "A":
        return [_ACT_CODE[r["act"]] for r in acts]
    if gran == "SA":
        return [_ST_CODE[r["street"]] + _ACT_CODE[r["act"]] for r in acts]
    if gran == "AZ":
        out = []
        for r in acts:
            c = _ACT_CODE[r["act"]]
            out.append(c + _size_bucket(r["ratio"], cuts) if r["act"] in _SIZED else c)
        return out
    if gran == "RA":
        cls = _position_classes(acts)
        return [("H" if r["hero"] else "V") + cls.get(r["seat"], "M")
                + _ACT_CODE[r["act"]] for r in acts]
    if gran == "HR":
        return [_ST_CODE[r["street"]] + _ACT_CODE[r["act"]] for r in acts if r["hero"]]
    return []


def _ngrams(tokens, n):
    if n <= 1:
        return list(tokens)
    if len(tokens) < n:
        return []
    return [">".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _chunk_size_cuts(hands_acts):
    """(q33,q66) of amount/pot_before pooled over the request's own hands.

    The only place a bet size enters the bank, and it enters as a within-chunk
    quantile, so any monotone shift of the sizing regime cancels exactly.
    """
    vals = [r["ratio"] for acts in hands_acts for r in acts if r["ratio"] is not None]
    if len(vals) < 6:
        return None
    # inverted_cdf (the empirical quantile) is EXACTLY invariant to replicating
    # the sample, so doubling the hand count cannot move the bucket boundaries.
    q = np.quantile(np.asarray(vals, dtype=np.float64), [1.0 / 3.0, 2.0 / 3.0],
                    method="inverted_cdf")
    lo, hi = float(q[0]), float(q[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return None
    return (lo, hi)


# ---------------------------------------------------------------------------
# frozen vocabulary + frozen live drop list
# ---------------------------------------------------------------------------


def _load_vocab(path=VOCAB_PATH):
    try:
        raw = json.load(open(path))
    except Exception:
        return {}
    v = {}
    for key, entry in raw.get("blocks", {}).items():
        g, n = key.rsplit("_", 1)
        v[(g, int(n))] = list(entry.get("vocab", []))
    return v


def _load_drop(path=DROP_PATH):
    try:
        return set(json.load(open(path)).get("drop", []))
    except Exception:
        return set()


_VOCAB = _load_vocab()
_DROP = _load_drop()

# prefix grouping for the conditional view, precomputed once
_PREFIX = {}
for _b in _BLOCKS:
    _g, _n = _b
    if _n >= 2:
        _PREFIX[_b] = [t.rsplit(">", 1)[0] for t in _VOCAB.get(_b, [])]

# ---------------------------------------------------------------------------
# per-block summary statistics (all computed in mix-normalized space)
# ---------------------------------------------------------------------------

_SUM_N1 = ("hand_repeat_pair_rate", "hand_js_mean", "hand_js_std", "hand_js_p90")
_SUM_N2 = ("cond_entropy_w", "cond_top1_w", "cond_entropy_std", "cond_nmi")


def _all_names():
    names = []
    for g, n in _BLOCKS:
        key = "%s_%d" % (g, n)
        for t in _VOCAB.get((g, n), []):
            names.append("%s__%s__%s" % ("rk" if n == 1 else "cp", key, t))
        for s in (_SUM_N1 if n == 1 else _SUM_N2):
            names.append("sum__%s__%s" % (key, s))
    return names


_ALL_NAMES = _all_names()
_KEEP = [i for i, nm in enumerate(_ALL_NAMES) if nm not in _DROP]
FEATURE_NAMES = [_ALL_NAMES[i] for i in _KEEP]
N_FEATURES = len(FEATURE_NAMES)
_ZEROS = [0.0] * N_FEATURES


def _build_groups():
    groups = {}
    for g, n in _BLOCKS:
        key = "%s_%d" % (g, n)
        groups["ngram_" + key] = [
            i for i, nm in enumerate(FEATURE_NAMES)
            if nm.startswith(("rk__" + key + "__", "cp__" + key + "__"))
        ]
        groups["summary_" + key] = [
            i for i, nm in enumerate(FEATURE_NAMES) if nm.startswith("sum__" + key + "__")
        ]
    groups["ngram_all"] = [i for i, nm in enumerate(FEATURE_NAMES)
                           if nm.startswith(("rk__", "cp__"))]
    groups["rank_n1"] = [i for i, nm in enumerate(FEATURE_NAMES) if nm.startswith("rk__")]
    groups["cond_n2plus"] = [i for i, nm in enumerate(FEATURE_NAMES) if nm.startswith("cp__")]
    groups["summary_all"] = [i for i, nm in enumerate(FEATURE_NAMES) if nm.startswith("sum__")]
    return groups


GROUPS = _build_groups()


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _js(p, q):
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    q = np.clip(q, 1e-12, None); q = q / q.sum()
    m = 0.5 * (p + q)
    d = 0.5 * float((p * np.log(p / m)).sum()) + 0.5 * float((q * np.log(q / m)).sum())
    return math.sqrt(max(d, 0.0))


def _rank_vector(counts, vocab):
    """Within-block rank of each vocab token's share, scaled to [0,1].

    Uses only the ORDERING of the block composition, so it is immune to any
    monotone reweighting of the action mix between domains.
    """
    k = len(vocab)
    if k == 0:
        return []
    v = np.asarray([counts.get(t, 0) for t in vocab], dtype=np.float64)
    order = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(np.float64)
    return list(order / max(k - 1, 1))


def _cond_vector(counts, vocab, prefixes, full_counts):
    """Shrunken P(last token | prefix) for every vocab n-gram.

    The denominator is the TOTAL count of the prefix over all continuations
    observed in the chunk (not just vocab ones), so a vocab-singleton prefix
    does not degenerate to a constant 1.0.

    Rare contexts (e.g. turn-call -> river-check) are seen only a handful of
    times even in a 90-hand chunk, so the raw ratio jumps in coarse discrete
    steps and is dominated by sampling noise.  Each estimate is therefore shrunk
    toward the chunk's OWN marginal for that continuation with ALPHA pseudo-
    counts: contexts with plenty of support are essentially unchanged, sparse
    ones fall back to the marginal instead of emitting noise.
    """
    if not vocab:
        return [], {}
    pref_tot = Counter()
    nxt_tot = Counter()
    for t, c in full_counts.items():
        pre, nx = t.rsplit(">", 1)
        pref_tot[pre] += c
        nxt_tot[nx] += c
    grand = float(sum(nxt_tot.values()))
    # alpha proportional to mean prefix support -> exactly replication invariant
    alpha = _COND_ALPHA_REL * grand / max(len(pref_tot), 1)
    out = []
    for t, p in zip(vocab, prefixes):
        d = pref_tot.get(p, 0)
        m = (nxt_tot.get(t.rsplit(">", 1)[1], 0) / grand) if grand > 0 else 0.0
        out.append((counts.get(t, 0) + alpha * m) / (d + alpha)
                   if (d + alpha) > 0 else 0.0)
    return out, pref_tot


def _cond_summaries(full_counts, pref_tot):
    """Sequence-predictability statistics of the conditional transition matrix.

    Bot policies are more deterministic given context; these measure exactly
    that, and because they are conditional they do not move with the action mix.
    """
    if not full_counts or not pref_tot:
        return [0.0] * len(_SUM_N2)
    by_pref = {}
    for t, c in full_counts.items():
        by_pref.setdefault(t.rsplit(">", 1)[0], []).append(c)
    grand = float(sum(full_counts.values())) or 1.0
    ents, tops, ws = [], [], []
    for p, cs in by_pref.items():
        v = np.asarray(cs, dtype=np.float64)
        tot = v.sum()
        if tot <= 0:
            continue
        pr = v / tot
        k = max(len(pr), 2)
        ents.append(_entropy(pr) / math.log(k))
        tops.append(float(pr.max()))
        ws.append(tot / grand)
    if not ws:
        return [0.0] * len(_SUM_N2)
    w = np.asarray(ws); w = w / w.sum()
    e = np.asarray(ents); t1 = np.asarray(tops)
    ew = float((w * e).sum())
    estd = float(math.sqrt(max((w * (e - ew) ** 2).sum(), 0.0)))
    # normalized mutual information between prefix and next token
    nxt = Counter()
    for t, c in full_counts.items():
        nxt[t.rsplit(">", 1)[1]] += c
    pn = np.asarray(list(nxt.values()), dtype=np.float64); pn = pn / pn.sum()
    hn = _entropy(pn)
    hc = 0.0
    for p, cs in by_pref.items():
        v = np.asarray(cs, dtype=np.float64)
        hc += (v.sum() / grand) * _entropy(v / v.sum())
    nmi = (hn - hc) / hn if hn > 1e-9 else 0.0
    return [ew, float((w * t1).sum()), estd, float(nmi)]


def _hand_summaries(per_hand_tokens, vocab):
    """Per-hand statistics: repeat rate and hand-vs-chunk heterogeneity.

    Averages over hands, so chunk length (40 vs 100 hands) cannot move them.
    Heterogeneity is measured against THIS chunk's own pooled distribution --
    the within-request "chunk-pool average" -- not against a training prior,
    which is what keeps it domain-normalized.
    """
    if not vocab:
        return [0.0] * len(_SUM_N1)
    idx = {t: i for i, t in enumerate(vocab)}
    k = len(vocab)
    pool = np.zeros(k, dtype=np.float64)
    rows, rep, nrep = [], 0.0, 0
    for toks in per_hand_tokens:
        if not toks:
            continue
        if len(toks) >= 2:
            rep += sum(1 for i in range(len(toks) - 1)
                       if toks[i] == toks[i + 1]) / float(len(toks) - 1)
            nrep += 1
        v = np.zeros(k, dtype=np.float64)
        hit = 0
        for t in toks:
            j = idx.get(t)
            if j is not None:
                v[j] += 1.0
                hit += 1
        if hit > 0:
            pool += v
            rows.append(v / hit)
    rep = rep / nrep if nrep else 0.0
    if not rows or pool.sum() <= 0:
        return [rep, 0.0, 0.0, 0.0]
    pool = pool / pool.sum()
    js = np.asarray([_js(r, pool) for r in rows], dtype=np.float64)
    return [rep, float(js.mean()), float(js.std()),
            float(np.quantile(js, 0.90, method="inverted_cdf"))]


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def extract_features(chunk):
    """Fixed-arity live-robust feature vector for one chunk of hands.

    Never raises: malformed input yields a finite all-zero vector of the correct
    length.
    """
    try:
        return _extract(chunk)
    except Exception:
        return list(_ZEROS)


def _extract(chunk):
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        return list(_ZEROS)
    hands_acts = []
    for h in chunk:
        try:
            a = _hand_actions(h)
        except Exception:
            a = []
        if a:
            hands_acts.append(a)
    if not hands_acts:
        return list(_ZEROS)

    cuts = _chunk_size_cuts(hands_acts)
    tok = {}
    for g in sorted({g for g, _ in _BLOCKS}):
        seqs = []
        for acts in hands_acts:
            try:
                seqs.append(tokenize_hand(acts, g, cuts))
            except Exception:
                seqs.append([])
        tok[g] = seqs

    feats = []
    for g, n in _BLOCKS:
        per_hand = [_ngrams(s, n) for s in tok.get(g, [])]
        counts = Counter()
        for gr in per_hand:
            counts.update(gr)
        vocab = _VOCAB.get((g, n), [])
        if n == 1:
            feats.extend(_rank_vector(counts, vocab))
            feats.extend(_hand_summaries(per_hand, vocab))
        else:
            cv, pref_tot = _cond_vector(counts, vocab, _PREFIX.get((g, n), []), counts)
            feats.extend(cv)
            feats.extend(_cond_summaries(counts, pref_tot))

    if len(feats) != len(_ALL_NAMES):
        return list(_ZEROS)
    out = []
    for i in _KEEP:
        v = float(feats[i])
        out.append(v if math.isfinite(v) else 0.0)
    return out


def extract_matrix(chunks):
    if not chunks:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.asarray([extract_features(c) for c in chunks], dtype=np.float64)


# ---------------------------------------------------------------------------
# pd-coast within-request rank view
# ---------------------------------------------------------------------------


def within_batch_rank(X, method="average", scale=True):
    """Replace every column by its rank among the rows of THIS request, in [0,1].

    Invariant to any strictly monotone per-column shift between training and
    serving.  Constant columns map to 0.5; NaN/inf are imputed with the column
    median before ranking.
    """
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    if A.size == 0:
        return A.copy()
    A = np.where(np.isfinite(A), A, np.nan)
    if np.isnan(A).any():
        med = np.nanmedian(A, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        A = np.where(np.isnan(A), med[None, :], A)
    n, m = A.shape
    if n == 1:
        return np.full_like(A, 0.5)
    order = np.argsort(A, axis=0, kind="stable")
    ranks = np.empty_like(A)
    rows = np.arange(n, dtype=np.float64)
    for j in range(m):
        o = order[:, j]
        r = np.empty(n, dtype=np.float64)
        r[o] = rows
        if method == "average":
            col = A[o, j]
            i = 0
            while i < n:
                k = i
                while k + 1 < n and col[k + 1] == col[i]:
                    k += 1
                if k > i:
                    r[o[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
        ranks[:, j] = r
    if scale:
        ranks = ranks / float(n - 1)
    const = A.max(axis=0) == A.min(axis=0)
    if const.any():
        ranks[:, const] = 0.5
    return ranks
