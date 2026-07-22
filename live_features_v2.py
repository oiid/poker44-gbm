"""live_features_v2.py -- STRICT SUPERSET of the deployed 260-feature live bank.

Columns [0:260) are produced by calling the DEPLOYED
    /root/bittensor/Poker44-subnet/neurons/live_features.py::extract_features
verbatim, so they are bit-identical to what the miner serves today (asserted
programmatically by test_superset() / validate_v2_groups.py).  Everything from
column 260 on is ADDITIVE and grouped, so the search agent can ablate groups
without touching the frozen part.

WHAT THE ADDITIVE GROUPS ARE
----------------------------
They are ports of machinery found in the leaders' repos (intel_live/), rewritten
to obey THIS bank's design contract (no raw magnitudes, no vocabulary-size /
diversity leakage, size invariance, hand-order invariance, truncation
robustness).  Provenance per group:

  fa_sz        Family A (Cold, ranks 1-3) -- poker44_ml/features.py
               ::_hand_ngram_doc.  Their token is
                   street_initial + action_code + POT-RELATIVE size bucket
               with ABSOLUTE cut points (<0.4 s, <0.9 m, <1.5 p, else o; "0"
               when amount<=0).  We have street+action (SA) and action+bucket
               (AZ) separately but NEVER the joint street+action+size token, and
               our bucket is a within-chunk tercile rather than their absolute
               cut.  Emitted here through OUR normalization: within-block rank
               for n=1, shrunken conditional P(next|prefix) for n=2,3.

  fa_za        Same absolute size cut points WITHOUT the street, i.e. a direct
               A/B against the deployed AZ block whose only difference is
               absolute-vs-tercile bucketing.

  fa_pos       Family A's "pos<rel><act>" token, rel = (actor_seat -
               button_seat) % max_seats.  Their raw form is table-size leaky
               (benchmark is 100% 6-max, live is 6/7/8/9-max: measured
               6:7464 9:5150 8:3880 7:730 hands), so it is ported as
               P(action | relative-position THIRD), which divides out both the
               table size and the action mix.

  fa_rate_struct / fa_rate_mix
               Family A's "<condition>_hand_rate" family (schema_
               high_aggression_hand_rate, low_action_entropy_hand_rate,
               high_actor_entropy_hand_rate, long_action_hand_rate): the
               FRACTION OF HANDS whose per-hand statistic crosses a fixed
               threshold.  Threshold-crossing rates are per-hand means, hence
               size invariant.  Split into a structural half (entropy /
               switching / runs / hero engagement) and a mix half (fold / check
               / call / passive shares), because the live action mix is known to
               shift and we want the KS verdict on each half separately.

  fa_struct    Family A's per-hand-feature aggregation (_aggregate_feature:
               mean/std/min/max/q10/q50/q90 of every per-hand statistic),
               restricted to the magnitude-free statistics and with min/max
               dropped -- extreme order statistics move with chunk length
               (benchmark 31 hands/chunk vs live 86) for purely mechanical
               reasons.

  fa_sig       Family A's signature-concentration features plus pd-coast
               (Family B, detection_model/model_v3/features.py) template
               concentration/entropy and half_disagreement.  Deliberately
               EXCLUDES every *_unique_share / *_unique_rate: those are the
               quadrant-(ii) mirage columns that exploded live.

  fa_len       Family A's "len"/"nseats" tokens and long_action_hand_rate.
               Kept as its own group because live hands are truncated to 5-8
               visible actions while benchmark hands span 1-19, so this group is
               EXPECTED to fail the live-collapse gate; it is present so the
               ablation can prove that rather than assume it.

  fa_deep      Hand-level "deepest street reached" + per-street action share,
               the feature-space projection of the Set Transformer's hand_end /
               actions_per_street channels (Family A poker44_ml/
               sequence_model.py::encode_hand).  Derived from the ACTIONS, never
               from hand["streets"], which the validator empties live
               (measured: live streets-list length 0 for 7336 of 17224 hands).

ADMISSIBILITY (enforced by validate_v2_groups.py, not by this module)
  a group is admissible only if it adds ZERO features with
  KS(benchmark, newest live capture) >= 0.60.

stdlib + numpy only.  Never raises; malformed input yields finite zeros.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "GROUPS",
    "N_FEATURES",
    "N_BASE",
    "extract_features",
    "extract_matrix",
    "within_batch_rank",
    "VOCAB_V2_PATH",
    "LF",
]

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- the deployed bank, loaded from the SERVED path -------------------------
# Loading the served file (not the staging copy) is what makes "first 260
# columns are what production emits" true by construction rather than by
# convention.
_DEPLOYED_DIR = os.environ.get(
    "POKER44_LIVE_FEATURES_DIR", "/root/bittensor/Poker44-subnet/neurons"
)


def _load_deployed():
    path = os.path.join(_DEPLOYED_DIR, "live_features.py")
    spec = importlib.util.spec_from_file_location("live_features_deployed", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("live_features_deployed", mod)
    spec.loader.exec_module(mod)
    return mod


LF = _load_deployed()
N_BASE = LF.N_FEATURES
within_batch_rank = LF.within_batch_rank


def frozen_triple_hash():
    """sha256 over (live_features.py, ngram_vocab.json, live_drop_list.json)."""
    h = hashlib.sha256()
    for name in ("live_features.py", "ngram_vocab.json", "live_drop_list.json"):
        with open(os.path.join(_DEPLOYED_DIR, name), "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Family A token alphabet (verbatim cut points)
# ---------------------------------------------------------------------------

_FA_ACT = {"fold": "F", "call": "C", "raise": "R", "check": "K", "bet": "B"}
_FA_ST = {"preflop": "p", "flop": "f", "turn": "t", "river": "r"}
_STREET_IDX = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
_ACTS = ("fold", "check", "call", "bet", "raise")

# Family A poker44_ml/features.py::_hand_ngram_doc, verbatim:
#     bucket = "0"  if amount <= 0
#              "?"  if pot_before <= 0
#              "s"  if ratio < 0.40
#              "m"  if ratio < 0.90
#              "p"  if ratio < 1.50
#              "o"  otherwise
# LF._hand_actions already yields ratio=None exactly when amount<=0 or
# pot_before<=0, and pot_before>0 holds for 100% of actions in both domains
# (measured), so ratio is None iff their bucket would be "0".
_FA_CUTS = (0.40, 0.90, 1.50)
_FA_BUCKETS = ("s", "m", "p", "o")

_EXT_BLOCKS = (("SZ", 1), ("SZ", 2), ("SZ", 3), ("ZA", 1), ("ZA", 2))

VOCAB_V2_PATH = os.path.join(_HERE, "ngram_vocab_v2.json")

_POS = ("E", "M", "L")
_COND_ALPHA_REL = 0.3
_RATE_THRESHOLDS = (0.20, 0.35, 0.50, 0.65)
_STRUCT_RATE_STATS = (
    "aggression",
    "action_entropy",
    "actor_entropy",
    "switch_rate",
    "action_run_max_share",
    "hero_action_share",
    "postflop_share",
)
_MIX_RATE_STATS = ("fold_share", "check_share", "call_share", "passivity")
_STRUCT_STATS = (
    "aggression",
    "passivity",
    "action_entropy",
    "actor_entropy",
    "street_entropy",
    "switch_rate",
    "action_run_max_share",
    "actor_run_max_share",
    "pot_monotonic_rate",
    "hero_action_share",
    "nonzero_amount_share",
)
_AGGS = ("mean", "std", "q10", "q50", "q90")


def _abs_bucket(ratio):
    if ratio is None:
        return "0"
    if ratio < _FA_CUTS[0]:
        return _FA_BUCKETS[0]
    if ratio < _FA_CUTS[1]:
        return _FA_BUCKETS[1]
    if ratio < _FA_CUTS[2]:
        return _FA_BUCKETS[2]
    return _FA_BUCKETS[3]


def tokenize_hand_v2(acts, gran):
    """Token sequence of one hand for the v2 blocks (no chunk-level state)."""
    if not acts:
        return []
    if gran == "SZ":
        return [_FA_ST.get(r["street"], "x") + _FA_ACT[r["act"]] + _abs_bucket(r["ratio"])
                for r in acts]
    if gran == "ZA":
        return [_FA_ACT[r["act"]] + _abs_bucket(r["ratio"]) for r in acts]
    return []


# ---------------------------------------------------------------------------
# frozen v2 vocabulary
# ---------------------------------------------------------------------------


def _load_vocab_v2(path=VOCAB_V2_PATH):
    try:
        raw = json.load(open(path))
    except Exception:
        return {}
    v = {}
    for key, entry in raw.get("blocks", {}).items():
        g, n = key.rsplit("_", 1)
        v[(g, int(n))] = list(entry.get("vocab", []))
    return v


_VOCAB2 = _load_vocab_v2()
_PREFIX2 = {}
for _b in _EXT_BLOCKS:
    if _b[1] >= 2:
        _PREFIX2[_b] = [t.rsplit(">", 1)[0] for t in _VOCAB2.get(_b, [])]


# ---------------------------------------------------------------------------
# per-hand parsing (extra fields the deployed bank does not expose)
# ---------------------------------------------------------------------------


def _f(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _hand_rows(hand):
    """(acts, extras) for one hand.  acts is LF's own normalized action list."""
    acts = LF._hand_actions(hand)
    if not acts:
        return None
    meta = hand.get("metadata") if isinstance(hand, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    try:
        max_seats = int(meta.get("max_seats") or 0)
    except (TypeError, ValueError):
        max_seats = 0
    try:
        button = int(meta.get("button_seat") or 0)
    except (TypeError, ValueError):
        button = 0
    players = hand.get("players") if isinstance(hand, dict) else None
    n_players = len(players) if isinstance(players, list) else 0
    pot_after = []
    raw = hand.get("actions") if isinstance(hand, dict) else None
    if isinstance(raw, list):
        for a in raw:
            if isinstance(a, dict) and a.get("action_type") in LF._ACT_CODE:
                pot_after.append(_f(a.get("pot_after"), 0.0))
    if len(pot_after) != len(acts):
        pot_after = [0.0] * len(acts)
    return acts, {
        "max_seats": max_seats,
        "button": button,
        "n_players": n_players,
        "pot_after": pot_after,
    }


def _norm_entropy(values):
    """Family A poker44_ml/features.py::_entropy -- entropy / log(#distinct)."""
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    if total <= 0.0 or len(counts) <= 1:
        return 0.0
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent / math.log(len(counts))


def _max_run_share(values):
    if not values:
        return 0.0
    longest = 1
    cur = 1
    for a, b in zip(values, values[1:]):
        if a == b:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest / float(len(values))


def _hand_stats(acts, extra):
    """Magnitude-free per-hand statistics."""
    n = float(len(acts))
    types = [r["act"] for r in acts]
    seats = [r["seat"] for r in acts]
    streets = [r["street"] for r in acts]
    c = Counter(types)
    aggressive = c.get("bet", 0) + c.get("raise", 0)
    passive = c.get("call", 0) + c.get("check", 0)
    pa = extra["pot_after"]
    monotone = sum(1 for a, b in zip(pa, pa[1:]) if b + 1e-9 >= a)
    hero = sum(1 for r in acts if r["hero"])
    postflop = sum(1 for s in streets if s != "preflop")
    distinct = len({s for s in seats if s is not None})
    return {
        "n_act": n,
        "n_actors": float(distinct),
        "aggression": aggressive / n,
        "passivity": passive / n,
        "fold_share": c.get("fold", 0) / n,
        "check_share": c.get("check", 0) / n,
        "call_share": c.get("call", 0) / n,
        "action_entropy": _norm_entropy(types),
        "actor_entropy": _norm_entropy([str(s) for s in seats]),
        "street_entropy": _norm_entropy(streets),
        "switch_rate": (sum(1 for a, b in zip(seats, seats[1:]) if a != b)
                        / max(len(seats) - 1, 1)) if len(seats) > 1 else 0.0,
        "action_run_max_share": _max_run_share(types),
        "actor_run_max_share": _max_run_share([str(s) for s in seats]),
        "pot_monotonic_rate": monotone / max(len(pa) - 1, 1) if len(pa) > 1 else 0.0,
        "hero_action_share": hero / n,
        "nonzero_amount_share": sum(1 for r in acts if r["ratio"] is not None) / n,
        "postflop_share": postflop / n,
        "unique_actor_share": (distinct / extra["n_players"]) if extra["n_players"] else 0.0,
        "n_players": float(extra["n_players"]),
        "deepest": max((_STREET_IDX.get(s, 0) for s in streets), default=0),
    }


def _pos_third(seat, button, max_seats):
    if seat is None or max_seats <= 0:
        return None
    try:
        rel = (int(seat) - int(button)) % int(max_seats)
    except (TypeError, ValueError):
        return None
    u = (rel + 0.5) / float(max_seats)
    return _POS[0] if u < 1.0 / 3 else (_POS[1] if u < 2.0 / 3 else _POS[2])


def _q(v, p):
    if not len(v):
        return 0.0
    return float(np.quantile(np.asarray(v, dtype=np.float64), p, method="inverted_cdf"))


def _agg5(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return [0.0] * 5
    return [float(a.mean()), float(a.std()), _q(a, 0.10), _q(a, 0.50), _q(a, 0.90)]


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def _ext_names():
    names = []
    grp = {}

    def add(group, items):
        start = len(names)
        names.extend(items)
        grp.setdefault(group, []).extend(range(start, len(names)))

    for g, n in _EXT_BLOCKS:
        key = "%s_%d" % (g, n)
        group = "fa_sz" if g == "SZ" else "fa_za"
        add(group, ["%s__%s__%s" % ("rk" if n == 1 else "cp", key, t)
                    for t in _VOCAB2.get((g, n), [])])
        add(group, ["sum__%s__%s" % (key, s)
                    for s in (LF._SUM_N1 if n == 1 else LF._SUM_N2)])

    add("fa_pos", ["pos__P%s__%s" % (p, a) for p in _POS for a in _ACTS])
    add("fa_rate_struct", ["rate__%s__ge%02d" % (s, int(t * 100))
                           for s in _STRUCT_RATE_STATS for t in _RATE_THRESHOLDS])
    add("fa_rate_mix", ["rate__%s__ge%02d" % (s, int(t * 100))
                        for s in _MIX_RATE_STATS for t in _RATE_THRESHOLDS])
    add("fa_struct", ["st__%s__%s" % (s, a) for s in _STRUCT_STATS for a in _AGGS])
    add("fa_sig", ["sig__action_top_share", "sig__actor_top_share",
                   "sig__street_top_share", "sig__sizebucket_top_share",
                   "sig__template_concentration", "sig__template_entropy",
                   "sig__half_disagreement"])
    add("fa_len", ["len__n_act__%s" % a for a in _AGGS]
        + ["len__n_actors__%s" % a for a in _AGGS]
        + ["len__rate_n_act_ge%d" % k for k in (4, 8, 12)]
        + ["len__unique_actor_share_mean", "len__n_players_mean"])
    add("fa_deep", ["deep__end_%s" % s for s in ("preflop", "flop", "turn", "river")]
        + ["deep__actshare_%s" % s for s in ("preflop", "flop", "turn", "river")]
        + ["deep__any_postflop_rate"])
    return names, grp


_EXT_NAMES, _EXT_GROUPS = _ext_names()
N_EXT = len(_EXT_NAMES)
_EXT_ZEROS = [0.0] * N_EXT

FEATURE_NAMES = list(LF.FEATURE_NAMES) + list(_EXT_NAMES)
N_FEATURES = len(FEATURE_NAMES)

GROUPS = {("base__" + k): list(v) for k, v in LF.GROUPS.items()}
GROUPS["base260"] = list(range(N_BASE))
for _g, _idx in _EXT_GROUPS.items():
    GROUPS[_g] = [N_BASE + i for i in _idx]
GROUPS["v2_all_new"] = [N_BASE + i for i in range(N_EXT)]

EXT_GROUP_ORDER = ["fa_sz", "fa_za", "fa_pos", "fa_rate_struct", "fa_rate_mix",
                   "fa_struct", "fa_sig", "fa_len", "fa_deep"]


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def _cond_simple(joint, marg_key_totals, next_totals, keys, alpha_rel=_COND_ALPHA_REL):
    """P(next | key) shrunk toward the pooled next-marginal.

    Same shrinkage discipline as LF._cond_vector: the pseudo-count scales with
    the mean support so the transform is invariant to replicating the chunk.
    """
    grand = float(sum(next_totals.values()))
    if grand <= 0:
        return [0.0] * (len(keys) * len(_ACTS))
    alpha = alpha_rel * grand / max(len(marg_key_totals), 1)
    out = []
    for k in keys:
        d = marg_key_totals.get(k, 0)
        for a in _ACTS:
            m = next_totals.get(a, 0) / grand
            out.append((joint.get((k, a), 0) + alpha * m) / (d + alpha)
                       if (d + alpha) > 0 else 0.0)
    return out


def _extract_ext(chunk):
    if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
        return list(_EXT_ZEROS)
    parsed = []
    for h in chunk:
        try:
            r = _hand_rows(h)
        except Exception:
            r = None
        if r:
            parsed.append(r)
    if not parsed:
        return list(_EXT_ZEROS)

    feats = []

    # ---- fa_sz / fa_za : Family A token alphabet through our normalization ---
    tok = {}
    for g in ("SZ", "ZA"):
        tok[g] = [tokenize_hand_v2(a, g) for a, _ in parsed]
    for g, n in _EXT_BLOCKS:
        per_hand = [LF._ngrams(s, n) for s in tok[g]]
        counts = Counter()
        for gr in per_hand:
            counts.update(gr)
        vocab = _VOCAB2.get((g, n), [])
        if n == 1:
            feats.extend(LF._rank_vector(counts, vocab))
            feats.extend(LF._hand_summaries(per_hand, vocab))
        else:
            cv, pref_tot = LF._cond_vector(counts, vocab, _PREFIX2.get((g, n), []), counts)
            feats.extend(cv)
            feats.extend(LF._cond_summaries(counts, pref_tot))

    # ---- fa_pos : P(action | relative-position third) -----------------------
    joint = Counter()
    pos_tot = Counter()
    act_tot = Counter()
    for acts, extra in parsed:
        for r in acts:
            p = _pos_third(r["seat"], extra["button"], extra["max_seats"])
            if p is None:
                continue
            joint[(p, r["act"])] += 1
            pos_tot[p] += 1
            act_tot[r["act"]] += 1
    feats.extend(_cond_simple(joint, pos_tot, act_tot, _POS))

    # ---- per-hand statistics -------------------------------------------------
    stats = [_hand_stats(acts, extra) for acts, extra in parsed]
    nh = float(len(stats))

    for name in _STRUCT_RATE_STATS:
        col = [s[name] for s in stats]
        for t in _RATE_THRESHOLDS:
            feats.append(sum(1 for v in col if v >= t) / nh)
    for name in _MIX_RATE_STATS:
        col = [s[name] for s in stats]
        for t in _RATE_THRESHOLDS:
            feats.append(sum(1 for v in col if v >= t) / nh)
    for name in _STRUCT_STATS:
        feats.extend(_agg5([s[name] for s in stats]))

    # ---- fa_sig : concentration only, never diversity ------------------------
    act_sig = Counter()
    actor_sig = Counter()
    street_sig = Counter()
    size_sig = Counter()
    templates = Counter()
    for (acts, _extra), s in zip(parsed, stats):
        act_sig[tuple(r["act"] for r in acts)] += 1
        actor_sig[tuple(str(r["seat"]) for r in acts)] += 1
        street_sig[tuple(r["street"] for r in acts)] += 1
        size_sig[tuple(_abs_bucket(r["ratio"]) for r in acts)] += 1
        templates[tuple(round(s[k], 2) for k in
                        ("fold_share", "check_share", "call_share", "aggression"))] += 1
    feats.append(max(act_sig.values()) / nh)
    feats.append(max(actor_sig.values()) / nh)
    feats.append(max(street_sig.values()) / nh)
    feats.append(max(size_sig.values()) / nh)
    feats.append(max(templates.values()) / nh)
    feats.append(_norm_entropy(list(templates.elements())))
    # pd-coast half_disagreement, restricted to the magnitude-free stat vector.
    if len(stats) >= 4:
        M = np.asarray([[s[k] for k in _STRUCT_STATS] for s in stats], dtype=np.float64)
        sigs = np.asarray(
            [int.from_bytes(hashlib.sha256(np.round(row, 4).tobytes()).digest()[:8], "big")
             for row in M], dtype=np.uint64)
        order = np.argsort(sigs, kind="stable")
        left, right = M[order[::2]], M[order[1::2]]
        m = min(len(left), len(right))
        feats.append(float(np.mean(np.abs(left[:m].mean(0) - right[:m].mean(0)))))
    else:
        feats.append(0.0)

    # ---- fa_len -------------------------------------------------------------
    feats.extend(_agg5([s["n_act"] for s in stats]))
    feats.extend(_agg5([s["n_actors"] for s in stats]))
    for k in (4, 8, 12):
        feats.append(sum(1 for s in stats if s["n_act"] >= k) / nh)
    feats.append(float(np.mean([s["unique_actor_share"] for s in stats])))
    feats.append(float(np.mean([s["n_players"] for s in stats])))

    # ---- fa_deep ------------------------------------------------------------
    for i in range(4):
        feats.append(sum(1 for s in stats if s["deepest"] == i) / nh)
    share = np.zeros(4, dtype=np.float64)
    for acts, _extra in parsed:
        n = float(len(acts))
        c = Counter(_STREET_IDX.get(r["street"], 0) for r in acts)
        for i in range(4):
            share[i] += c.get(i, 0) / n
    feats.extend((share / nh).tolist())
    feats.append(sum(1 for s in stats if s["postflop_share"] > 0) / nh)

    if len(feats) != N_EXT:
        return list(_EXT_ZEROS)
    return [float(v) if math.isfinite(float(v)) else 0.0 for v in feats]


def extract_features(chunk):
    """260 deployed columns (bit-identical) followed by the additive groups."""
    base = LF.extract_features(chunk)
    try:
        ext = _extract_ext(chunk)
    except Exception:
        ext = list(_EXT_ZEROS)
    if len(ext) != N_EXT:
        ext = list(_EXT_ZEROS)
    return list(base) + ext


def extract_matrix(chunks):
    if not chunks:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.asarray([extract_features(c) for c in chunks], dtype=np.float64)


MANIFEST_PATH = os.path.join(_HERE, "live_features_v2_manifest.json")


def load_groups(path=MANIFEST_PATH):
    """Group -> column indices, including the KS-pruned ``*_ks`` and the
    pruned-and-deduplicated ``*_novel`` variants written by
    validate_v2_groups.py / prune_v2_groups.py.  Falls back to the in-module
    GROUPS when the manifest has not been built yet."""
    try:
        man = json.load(open(path))
    except Exception:
        return dict(GROUPS)
    out = dict(GROUPS)
    for g, e in man.get("groups", {}).items():
        out[g] = list(e.get("indices", []))
    return out


def test_superset(chunks):
    """Assert columns [0:260) equal the deployed bank bit-for-bit."""
    ok = 0
    for c in chunks:
        a = np.asarray(LF.extract_features(c), dtype=np.float64)
        b = np.asarray(extract_features(c)[:N_BASE], dtype=np.float64)
        if not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
            raise AssertionError("column drift at chunk %d" % ok)
        ok += 1
    return ok
