"""Chunk feature extraction v3 for Poker44 bot detection (STAGING).

Append-only extension of chunk_features_v2: the first 148 values are
produced by the EXACT v2 code path (111 deployed v1 features + 37
signature/dispersion features, bit-identical), followed by the "coherent
block" ported from the leader repo (Evil-DrPork/pd-coast,
detection_model/model_v4/features.py): per-hand behavioral scalars
(pot evolution in bb, stack geometry, action-sequence entropy and
run-lengths, actor switch rate, hero shares, raise_to/call_to stats,
seat geometry) each summarized across the chunk's hands with 8
distribution stats (mean/std/mad/q10/q25/q50/q75/q90), plus their
6-signature-kinds x 6-stats grid (action / actor / street / amount /
street_action / full x top1_share / top2_share / unique_rate /
singleton_share / entropy / repeat_pair_rate).

Adaptations to OUR serve reality (miner-visible sanitized payloads, see
poker44/validator/payload_view.py):
  * the sanitizer forces button_seat=0 and strips blind/ante/all_in
    actions, so the leader's 4 button-geometry scalars and their
    blind/allin shares are dead here and are not computed;
  * candidates that are (near-)constant on live captures, or duplicates
    of an existing column, are dropped from the emitted vector
    (``V3_KEPT`` below, baked from prune_v3.py output).

Invariants:
  * append-only: FEATURE_NAMES[:148] == chunk_features_v2.FEATURE_NAMES
    and the values are bit-identical (same code path);
  * hand-order invariant: every cross-hand reduction sorts first
    (scalar lists ascending, signature counts descending), so any
    permutation of the same hands yields the bit-identical vector;
  * size-invariant: per-hand scalars do not scale with the number of
    hands in the chunk; cross-hand summaries are distribution stats;
  * never raises: malformed hands contribute zero rows / empty
    signatures; any failure returns a zero vector.

Stdlib-only, like v1/v2.
"""

from __future__ import annotations

import math
import sys
from bisect import bisect_right
from pathlib import Path

try:
    from neurons.chunk_features_v2 import (  # noqa: F401
        FEATURE_NAMES as V2_FEATURE_NAMES,
        N_FEATURES_V1,
        _build_sig,
        _build_v1,
        _safe_float,
    )
except ImportError:  # staging use before the files land in neurons/
    _here = str(Path(__file__).resolve().parent)
    for _p in (_here, "/root/bittensor/Poker44-subnet"):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from chunk_features_v2 import (  # noqa: F401
        FEATURE_NAMES as V2_FEATURE_NAMES,
        N_FEATURES_V1,
        _build_sig,
        _build_v1,
        _safe_float,
    )

N_FEATURES_V2 = len(V2_FEATURE_NAMES)

# ---------------------------------------------------------------------------
# coherent block definitions (leader port, sanitizer-aware)
# ---------------------------------------------------------------------------

_AMOUNT_BOUNDS = (0.0, 0.25, 0.50, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
_AGGRESSIVE = ("bet", "raise")
_PASSIVE = ("check", "call")

_DIST_STATS = ("mean", "std", "mad", "q10", "q25", "q50", "q75", "q90")

_HAND_SCALARS = (
    # Pot evolution, in big blinds.
    "pot_before_mean_bb", "pot_before_max_bb", "pot_after_mean_bb",
    "pot_after_max_bb", "pot_after_final_bb", "pot_change_abs_mean_bb",
    "pot_delta_positive_mean_bb", "pot_growth_bb", "pot_monotonic_rate",
    # Public starting-stack geometry, in big blinds.
    "stack_mean_bb", "stack_std_bb", "stack_range_bb", "hero_stack_bb",
    "hero_stack_to_mean",
    # Action-sequence complexity.
    "action_count", "action_type_unique", "actor_unique", "street_unique",
    "actor_switch_rate", "action_run_max_share", "actor_run_max_share",
    "action_entropy", "actor_entropy", "street_entropy",
    "preflop_share", "postflop_share", "aggressive_share", "passive_share",
    "amount_mean_bb", "amount_std_bb", "amount_q90_bb", "amount_max_bb",
    "amount_nonzero_share",
    # Public seat geometry (button fields are dead post-sanitizer).
    "player_count", "seat_utilization", "hero_seat_norm",
    # Hero behavior.
    "hero_action_count", "hero_action_share", "hero_aggressive_share",
    "hero_fold_share",
    # Explicit target amounts, in big blinds.
    "raise_to_count", "raise_to_share", "raise_to_mean_bb", "raise_to_max_bb",
    "call_to_count", "call_to_share", "call_to_mean_bb", "call_to_max_bb",
)
_N_SCALARS = len(_HAND_SCALARS)

_SIG_KINDS = ("action", "actor", "street", "amount", "street_action", "full")
_SIG_STATS = ("top1_share", "top2_share", "unique_rate", "singleton_share",
              "entropy", "repeat_pair_rate")

V3_CANDIDATE_NAMES = tuple(
    [f"ch__{name}__{stat}" for name in _HAND_SCALARS for stat in _DIST_STATS]
    + [f"cs__{kind}__{stat}" for kind in _SIG_KINDS for stat in _SIG_STATS]
)

# Baked by prune_v3.py: candidates that survive the live-capture
# (near-)constant filter and the duplicate filter.  ``None`` means
# "keep every candidate" (pre-pruning bootstrap mode only).
# --- BEGIN V3_KEPT (generated by prune_v3.py; do not edit by hand) ---
V3_KEPT = (
    "ch__pot_before_mean_bb__mean",
    "ch__pot_before_mean_bb__std",
    "ch__pot_before_mean_bb__mad",
    "ch__pot_before_mean_bb__q10",
    "ch__pot_before_mean_bb__q25",
    "ch__pot_before_mean_bb__q50",
    "ch__pot_before_mean_bb__q75",
    "ch__pot_before_mean_bb__q90",
    "ch__pot_before_max_bb__mean",
    "ch__pot_before_max_bb__std",
    "ch__pot_before_max_bb__mad",
    "ch__pot_before_max_bb__q10",
    "ch__pot_before_max_bb__q25",
    "ch__pot_before_max_bb__q50",
    "ch__pot_before_max_bb__q75",
    "ch__pot_after_mean_bb__mean",
    "ch__pot_after_mean_bb__std",
    "ch__pot_after_mean_bb__mad",
    "ch__pot_after_mean_bb__q10",
    "ch__pot_after_mean_bb__q25",
    "ch__pot_after_mean_bb__q50",
    "ch__pot_after_mean_bb__q75",
    "ch__pot_after_mean_bb__q90",
    "ch__pot_after_max_bb__std",
    "ch__pot_after_max_bb__mad",
    "ch__pot_after_max_bb__q10",
    "ch__pot_after_max_bb__q25",
    "ch__pot_after_max_bb__q50",
    "ch__pot_after_max_bb__q75",
    "ch__pot_after_final_bb__mean",
    "ch__pot_after_final_bb__std",
    "ch__pot_after_final_bb__mad",
    "ch__pot_after_final_bb__q10",
    "ch__pot_after_final_bb__q25",
    "ch__pot_after_final_bb__q50",
    "ch__pot_after_final_bb__q75",
    "ch__pot_change_abs_mean_bb__mean",
    "ch__pot_change_abs_mean_bb__std",
    "ch__pot_change_abs_mean_bb__mad",
    "ch__pot_change_abs_mean_bb__q10",
    "ch__pot_change_abs_mean_bb__q25",
    "ch__pot_change_abs_mean_bb__q50",
    "ch__pot_change_abs_mean_bb__q75",
    "ch__pot_change_abs_mean_bb__q90",
    "ch__pot_delta_positive_mean_bb__mean",
    "ch__pot_delta_positive_mean_bb__std",
    "ch__pot_delta_positive_mean_bb__mad",
    "ch__pot_delta_positive_mean_bb__q10",
    "ch__pot_delta_positive_mean_bb__q25",
    "ch__pot_delta_positive_mean_bb__q50",
    "ch__pot_delta_positive_mean_bb__q75",
    "ch__pot_delta_positive_mean_bb__q90",
    "ch__pot_growth_bb__mean",
    "ch__pot_growth_bb__mad",
    "ch__pot_growth_bb__q10",
    "ch__pot_growth_bb__q25",
    "ch__pot_growth_bb__q50",
    "ch__pot_growth_bb__q75",
    "ch__pot_monotonic_rate__mean",
    "ch__pot_monotonic_rate__std",
    "ch__pot_monotonic_rate__mad",
    "ch__pot_monotonic_rate__q10",
    "ch__pot_monotonic_rate__q25",
    "ch__pot_monotonic_rate__q50",
    "ch__pot_monotonic_rate__q75",
    "ch__pot_monotonic_rate__q90",
    "ch__stack_mean_bb__mean",
    "ch__stack_mean_bb__std",
    "ch__stack_mean_bb__mad",
    "ch__stack_mean_bb__q10",
    "ch__stack_mean_bb__q25",
    "ch__stack_std_bb__mean",
    "ch__stack_std_bb__std",
    "ch__stack_std_bb__mad",
    "ch__stack_std_bb__q10",
    "ch__stack_std_bb__q25",
    "ch__stack_std_bb__q50",
    "ch__stack_std_bb__q75",
    "ch__stack_std_bb__q90",
    "ch__stack_range_bb__mean",
    "ch__stack_range_bb__std",
    "ch__stack_range_bb__mad",
    "ch__stack_range_bb__q10",
    "ch__stack_range_bb__q25",
    "ch__stack_range_bb__q50",
    "ch__stack_range_bb__q75",
    "ch__stack_range_bb__q90",
    "ch__hero_stack_bb__mad",
    "ch__hero_stack_bb__q10",
    "ch__hero_stack_bb__q25",
    "ch__hero_stack_bb__q50",
    "ch__hero_stack_to_mean__mean",
    "ch__hero_stack_to_mean__std",
    "ch__hero_stack_to_mean__mad",
    "ch__hero_stack_to_mean__q10",
    "ch__hero_stack_to_mean__q25",
    "ch__hero_stack_to_mean__q50",
    "ch__hero_stack_to_mean__q75",
    "ch__hero_stack_to_mean__q90",
    "ch__action_count__mad",
    "ch__action_count__q25",
    "ch__action_count__q50",
    "ch__action_count__q75",
    "ch__action_type_unique__mean",
    "ch__action_type_unique__std",
    "ch__action_type_unique__mad",
    "ch__action_type_unique__q10",
    "ch__action_type_unique__q25",
    "ch__action_type_unique__q75",
    "ch__actor_unique__mean",
    "ch__actor_unique__std",
    "ch__actor_unique__q10",
    "ch__actor_unique__q25",
    "ch__actor_unique__q75",
    "ch__actor_unique__q90",
    "ch__street_unique__mean",
    "ch__street_unique__std",
    "ch__street_unique__q25",
    "ch__street_unique__q50",
    "ch__street_unique__q90",
    "ch__actor_switch_rate__mad",
    "ch__actor_switch_rate__q10",
    "ch__actor_switch_rate__q25",
    "ch__actor_switch_rate__q50",
    "ch__action_run_max_share__mean",
    "ch__action_run_max_share__std",
    "ch__action_run_max_share__mad",
    "ch__action_run_max_share__q10",
    "ch__action_run_max_share__q25",
    "ch__action_run_max_share__q50",
    "ch__action_run_max_share__q75",
    "ch__action_run_max_share__q90",
    "ch__actor_run_max_share__mean",
    "ch__actor_run_max_share__std",
    "ch__actor_run_max_share__mad",
    "ch__actor_run_max_share__q10",
    "ch__actor_run_max_share__q25",
    "ch__actor_run_max_share__q50",
    "ch__actor_run_max_share__q75",
    "ch__actor_run_max_share__q90",
    "ch__action_entropy__mad",
    "ch__action_entropy__q10",
    "ch__action_entropy__q25",
    "ch__action_entropy__q50",
    "ch__action_entropy__q75",
    "ch__action_entropy__q90",
    "ch__actor_entropy__mean",
    "ch__actor_entropy__std",
    "ch__actor_entropy__mad",
    "ch__actor_entropy__q10",
    "ch__actor_entropy__q25",
    "ch__actor_entropy__q50",
    "ch__actor_entropy__q75",
    "ch__actor_entropy__q90",
    "ch__street_entropy__mean",
    "ch__street_entropy__std",
    "ch__street_entropy__mad",
    "ch__street_entropy__q25",
    "ch__street_entropy__q50",
    "ch__street_entropy__q75",
    "ch__street_entropy__q90",
    "ch__preflop_share__mean",
    "ch__preflop_share__std",
    "ch__preflop_share__mad",
    "ch__preflop_share__q10",
    "ch__preflop_share__q25",
    "ch__preflop_share__q50",
    "ch__preflop_share__q75",
    "ch__aggressive_share__mean",
    "ch__aggressive_share__std",
    "ch__aggressive_share__mad",
    "ch__aggressive_share__q50",
    "ch__aggressive_share__q75",
    "ch__aggressive_share__q90",
    "ch__passive_share__mean",
    "ch__passive_share__std",
    "ch__passive_share__mad",
    "ch__passive_share__q25",
    "ch__passive_share__q50",
    "ch__passive_share__q75",
    "ch__passive_share__q90",
    "ch__amount_mean_bb__mean",
    "ch__amount_mean_bb__std",
    "ch__amount_mean_bb__mad",
    "ch__amount_mean_bb__q10",
    "ch__amount_mean_bb__q25",
    "ch__amount_mean_bb__q50",
    "ch__amount_mean_bb__q75",
    "ch__amount_mean_bb__q90",
    "ch__amount_std_bb__mean",
    "ch__amount_std_bb__std",
    "ch__amount_std_bb__mad",
    "ch__amount_std_bb__q10",
    "ch__amount_std_bb__q25",
    "ch__amount_std_bb__q50",
    "ch__amount_std_bb__q75",
    "ch__amount_std_bb__q90",
    "ch__amount_q90_bb__mean",
    "ch__amount_q90_bb__std",
    "ch__amount_q90_bb__mad",
    "ch__amount_q90_bb__q10",
    "ch__amount_q90_bb__q25",
    "ch__amount_q90_bb__q50",
    "ch__amount_q90_bb__q75",
    "ch__amount_q90_bb__q90",
    "ch__amount_max_bb__mean",
    "ch__amount_max_bb__std",
    "ch__amount_max_bb__mad",
    "ch__amount_max_bb__q10",
    "ch__amount_max_bb__q25",
    "ch__amount_max_bb__q50",
    "ch__amount_max_bb__q75",
    "ch__amount_max_bb__q90",
    "ch__amount_nonzero_share__mean",
    "ch__amount_nonzero_share__std",
    "ch__amount_nonzero_share__mad",
    "ch__amount_nonzero_share__q10",
    "ch__amount_nonzero_share__q25",
    "ch__amount_nonzero_share__q50",
    "ch__amount_nonzero_share__q75",
    "ch__amount_nonzero_share__q90",
    "ch__player_count__std",
    "ch__player_count__mad",
    "ch__player_count__q10",
    "ch__player_count__q25",
    "ch__player_count__q50",
    "ch__player_count__q75",
    "ch__seat_utilization__mean",
    "ch__seat_utilization__std",
    "ch__hero_seat_norm__mean",
    "ch__hero_seat_norm__std",
    "ch__hero_seat_norm__mad",
    "ch__hero_seat_norm__q10",
    "ch__hero_seat_norm__q25",
    "ch__hero_seat_norm__q50",
    "ch__hero_seat_norm__q75",
    "ch__hero_seat_norm__q90",
    "ch__hero_action_count__q25",
    "ch__hero_action_count__q75",
    "ch__hero_action_count__q90",
    "ch__hero_action_share__mean",
    "ch__hero_action_share__std",
    "ch__hero_action_share__mad",
    "ch__hero_action_share__q25",
    "ch__hero_action_share__q50",
    "ch__hero_action_share__q75",
    "ch__hero_action_share__q90",
    "ch__hero_aggressive_share__mean",
    "ch__hero_aggressive_share__std",
    "ch__hero_aggressive_share__q90",
    "ch__hero_fold_share__mean",
    "ch__hero_fold_share__std",
    "ch__raise_to_count__mean",
    "ch__raise_to_count__std",
    "ch__raise_to_count__q75",
    "ch__raise_to_share__mean",
    "ch__raise_to_share__std",
    "ch__raise_to_share__q75",
    "ch__raise_to_share__q90",
    "ch__raise_to_mean_bb__mean",
    "ch__raise_to_mean_bb__std",
    "ch__raise_to_mean_bb__q75",
    "ch__raise_to_mean_bb__q90",
    "ch__raise_to_max_bb__mean",
    "ch__raise_to_max_bb__std",
    "ch__raise_to_max_bb__q75",
    "ch__raise_to_max_bb__q90",
    "ch__call_to_count__mean",
    "ch__call_to_count__std",
    "ch__call_to_count__q50",
    "ch__call_to_count__q75",
    "ch__call_to_count__q90",
    "ch__call_to_share__mean",
    "ch__call_to_share__std",
    "ch__call_to_share__mad",
    "ch__call_to_share__q25",
    "ch__call_to_share__q50",
    "ch__call_to_share__q75",
    "ch__call_to_share__q90",
    "ch__call_to_mean_bb__mean",
    "ch__call_to_mean_bb__std",
    "ch__call_to_mean_bb__mad",
    "ch__call_to_mean_bb__q25",
    "ch__call_to_mean_bb__q50",
    "ch__call_to_mean_bb__q75",
    "ch__call_to_mean_bb__q90",
    "ch__call_to_max_bb__mean",
    "ch__call_to_max_bb__std",
    "ch__call_to_max_bb__mad",
    "ch__call_to_max_bb__q50",
    "ch__call_to_max_bb__q75",
    "ch__call_to_max_bb__q90",
    "cs__action__top2_share",
    "cs__action__singleton_share",
    "cs__actor__top1_share",
    "cs__actor__top2_share",
    "cs__actor__unique_rate",
    "cs__actor__singleton_share",
    "cs__actor__entropy",
    "cs__actor__repeat_pair_rate",
    "cs__street__top1_share",
    "cs__street__top2_share",
    "cs__street__unique_rate",
    "cs__street__singleton_share",
    "cs__street__entropy",
    "cs__street__repeat_pair_rate",
    "cs__amount__top1_share",
    "cs__amount__top2_share",
    "cs__amount__unique_rate",
    "cs__amount__singleton_share",
    "cs__amount__entropy",
    "cs__amount__repeat_pair_rate",
    "cs__street_action__top2_share",
    "cs__street_action__singleton_share",
    "cs__full__top1_share",
    "cs__full__top2_share",
    "cs__full__unique_rate",
    "cs__full__singleton_share",
    "cs__full__entropy",
    "cs__full__repeat_pair_rate",
)
# --- END V3_KEPT ---


# ---------------------------------------------------------------------------
# helpers (stdlib, deterministic under hand permutation)
# ---------------------------------------------------------------------------

def _pos(v):
    """Bounded non-negative float for an untrusted JSON scalar."""
    x = _safe_float(v)
    if x < 0.0:
        return 0.0
    return x if x < 1_000_000.0 else 1_000_000.0


def _int0(v):
    try:
        return int(_safe_float(v))
    except (TypeError, ValueError, OverflowError):
        return 0


def _tok(v):
    """Canonical categorical token (leader style)."""
    if v is None:
        return "<missing>"
    s = str(v).strip().lower()
    return s[:48] if s else "<missing>"


def _q_sorted(s, q):
    """Linear-interpolation quantile of an ASCENDING-sorted list."""
    n = len(s)
    if n == 0:
        return 0.0
    pos = q * (n - 1)
    lo = int(pos)
    hi = lo + 1 if lo + 1 < n else n - 1
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _entropy_counts(counts):
    """Normalized entropy of positive counts; deterministic (sorted sum)."""
    k = len(counts)
    if k <= 1:
        return 0.0
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in sorted(counts):
        p = c / total
        h -= p * math.log(p + 1e-15)
    return h / math.log(k)


def _entropy_seq(seq):
    if len(seq) <= 1:
        return 0.0
    counts = {}
    for v in seq:
        counts[v] = counts.get(v, 0) + 1
    if len(counts) <= 1:
        return 0.0
    return _entropy_counts(list(counts.values()))


def _max_run_share(seq):
    if not seq:
        return 0.0
    longest = current = 1
    for prev, cur in zip(seq, seq[1:]):
        if cur == prev:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
    return longest / len(seq)


def _amt_bucket_coh(amount_bb):
    return bisect_right(_AMOUNT_BOUNDS, amount_bb if amount_bb > 0.0 else 0.0) - 1


_ZERO_ROW = (0.0,) * _N_SCALARS
_EMPTY_SIGS = {k: () for k in _SIG_KINDS}


def _hand_coherent(hand):
    """One hand -> (48-scalar row aligned to _HAND_SCALARS, signature dict)."""
    if not isinstance(hand, dict):
        return list(_ZERO_ROW), dict(_EMPTY_SIGS)
    meta = hand.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    raw_actions = hand.get("actions")
    actions = ([a for a in raw_actions if isinstance(a, dict)]
               if isinstance(raw_actions, list) else [])
    raw_players = hand.get("players")
    players = ([p for p in raw_players if isinstance(p, dict)]
               if isinstance(raw_players, list) else [])

    bb = abs(_safe_float(meta.get("bb"), 0.02))
    if bb <= 0.0:
        bb = 0.02
    hero_seat = _int0(meta.get("hero_seat"))
    player_seats = [_int0(p.get("seat")) for p in players]
    max_seats = max(
        1, _int0(meta.get("max_seats")),
        max((s for s in player_seats if s > 0), default=0), hero_seat,
    )

    a_types = [_tok(a.get("action_type")) for a in actions]
    a_streets = [_tok(a.get("street")) for a in actions]
    a_actors = [_int0(a.get("actor_seat")) for a in actions]
    amounts = []
    for a in actions:
        na = a.get("normalized_amount_bb")
        amt = _pos(na) if na is not None else _pos(a.get("amount")) / bb
        amounts.append(amt if amt < 1_000_000.0 else 1_000_000.0)
    a_buckets = [_amt_bucket_coh(x) for x in amounts]
    n_actions = len(actions)

    # ---- pot evolution ----------------------------------------------------
    pb = [_pos(a.get("pot_before")) / bb for a in actions]
    pa = [_pos(a.get("pot_after")) / bb for a in actions]
    if pb and pa:
        change_abs = sum(abs(y - x) for x, y in zip(pb, pa)) / n_actions
        delta_pos = sum(y - x if y > x else 0.0 for x, y in zip(pb, pa)) / n_actions
        growth = max(pa) - min(pb)
        if growth < 0.0:
            growth = 0.0
    else:
        change_abs = delta_pos = growth = 0.0
    if len(pa) > 1:
        mono = sum(1 for x, y in zip(pa, pa[1:]) if y >= x - 1e-9) / (len(pa) - 1)
    else:
        mono = 0.0

    # ---- stacks -----------------------------------------------------------
    stacks = [_pos(p.get("starting_stack")) / bb for p in players]
    if stacks:
        stack_mean = sum(stacks) / len(stacks)
        stack_std = math.sqrt(
            sum((x - stack_mean) ** 2 for x in stacks) / len(stacks))
        stack_range = max(stacks) - min(stacks)
    else:
        stack_mean = stack_std = stack_range = 0.0
    hero_stack = 0.0
    if hero_seat > 0:
        for p, seat in zip(players, player_seats):
            if seat == hero_seat:
                hero_stack = _pos(p.get("starting_stack")) / bb
                break

    # ---- action-sequence complexity ---------------------------------------
    nd = max(1, n_actions)
    if n_actions > 1:
        switch_rate = sum(
            1 for x, y in zip(a_actors, a_actors[1:]) if x != y) / (n_actions - 1)
    else:
        switch_rate = 0.0
    preflop_count = sum(1 for s in a_streets if s == "preflop")
    postflop_count = sum(
        1 for s in a_streets if s not in ("<missing>", "preflop"))
    if amounts:
        amt_sorted = sorted(amounts)
        amt_mean = sum(amt_sorted) / len(amt_sorted)
        amt_std = math.sqrt(
            sum((x - amt_mean) ** 2 for x in amt_sorted) / len(amt_sorted))
        amt_q90 = _q_sorted(amt_sorted, 0.90)
        amt_max = amt_sorted[-1]
        amt_nonzero = sum(1 for x in amounts if x > 0.0) / len(amounts)
    else:
        amt_mean = amt_std = amt_q90 = amt_max = amt_nonzero = 0.0

    # ---- hero behavior ----------------------------------------------------
    hero_types = ([t for t, s in zip(a_types, a_actors) if s == hero_seat]
                  if hero_seat > 0 else [])
    n_hero = len(hero_types)
    nh = max(1, n_hero)

    # ---- explicit targets -------------------------------------------------
    raise_targets = [_pos(a.get("raise_to")) / bb for a in actions
                     if a.get("raise_to") is not None]
    call_targets = [_pos(a.get("call_to")) / bb for a in actions
                    if a.get("call_to") is not None]

    row = [
        # pot
        sum(pb) / len(pb) if pb else 0.0,
        max(pb) if pb else 0.0,
        sum(pa) / len(pa) if pa else 0.0,
        max(pa) if pa else 0.0,
        pa[-1] if pa else 0.0,
        change_abs,
        delta_pos,
        growth,
        mono,
        # stacks
        stack_mean,
        stack_std,
        stack_range,
        hero_stack,
        hero_stack / stack_mean if hero_stack > 0.0 and stack_mean > 1e-6 else 0.0,
        # action-sequence complexity
        float(n_actions),
        float(len(set(a_types))),
        float(len({s for s in a_actors if s > 0})),
        float(len({s for s in a_streets if s != "<missing>"})),
        switch_rate,
        _max_run_share(a_types),
        _max_run_share(a_actors),
        _entropy_seq(a_types),
        _entropy_seq(a_actors),
        _entropy_seq(a_streets),
        preflop_count / nd,
        postflop_count / nd,
        sum(1 for t in a_types if t in _AGGRESSIVE) / nd,
        sum(1 for t in a_types if t in _PASSIVE) / nd,
        amt_mean,
        amt_std,
        amt_q90,
        amt_max,
        amt_nonzero,
        # seat geometry
        float(len(players)),
        len(players) / max_seats,
        hero_seat / max_seats if hero_seat > 0 else 0.0,
        # hero behavior
        float(n_hero),
        n_hero / nd,
        sum(1 for t in hero_types if t in _AGGRESSIVE) / nh,
        hero_types.count("fold") / nh,
        # explicit targets
        float(len(raise_targets)),
        len(raise_targets) / nd,
        sum(raise_targets) / len(raise_targets) if raise_targets else 0.0,
        max(raise_targets) if raise_targets else 0.0,
        float(len(call_targets)),
        len(call_targets) / nd,
        sum(call_targets) / len(call_targets) if call_targets else 0.0,
        max(call_targets) if call_targets else 0.0,
    ]
    sigs = {
        "action": tuple(a_types),
        "actor": tuple(a_actors),
        "street": tuple(a_streets),
        "amount": tuple(a_buckets),
        "street_action": tuple(zip(a_streets, a_types)),
        "full": tuple(zip(a_streets, a_actors, a_types, a_buckets)),
    }
    return row, sigs


def _coherent_values(chunk):
    """All V3 candidate values for one chunk, as {name: float}."""
    hands = chunk if isinstance(chunk, (list, tuple)) else []
    rows = []
    sig_lists = {k: [] for k in _SIG_KINDS}
    for h in hands:
        try:
            row, sigs = _hand_coherent(h)
        except Exception:  # noqa: BLE001 - malformed hand -> zero row
            row, sigs = list(_ZERO_ROW), dict(_EMPTY_SIGS)
        rows.append(row)
        for k in _SIG_KINDS:
            sig_lists[k].append(sigs[k])
    if not rows:
        rows = [list(_ZERO_ROW)]

    vals = {}
    # ---- per-hand scalar distributions (sorted -> order-invariant) --------
    n = len(rows)
    for j, name in enumerate(_HAND_SCALARS):
        xs = sorted(_safe_float(r[j]) for r in rows)
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / n)
        med = _q_sorted(xs, 0.50)
        dev = sorted(abs(x - med) for x in xs)
        prefix = f"ch__{name}__"
        vals[prefix + "mean"] = m
        vals[prefix + "std"] = sd
        vals[prefix + "mad"] = _q_sorted(dev, 0.50)
        vals[prefix + "q10"] = _q_sorted(xs, 0.10)
        vals[prefix + "q25"] = _q_sorted(xs, 0.25)
        vals[prefix + "q50"] = med
        vals[prefix + "q75"] = _q_sorted(xs, 0.75)
        vals[prefix + "q90"] = _q_sorted(xs, 0.90)

    # ---- cross-hand signature grid (sorted counts -> order-invariant) -----
    for kind in _SIG_KINDS:
        counts = {}
        for s in sig_lists[kind]:
            counts[s] = counts.get(s, 0) + 1
        cvals = sorted(counts.values(), reverse=True)
        total = sum(cvals)
        prefix = f"cs__{kind}__"
        if total <= 0:
            for stat in _SIG_STATS:
                vals[prefix + stat] = 0.0
            continue
        vals[prefix + "top1_share"] = cvals[0] / total
        vals[prefix + "top2_share"] = sum(cvals[:2]) / total
        vals[prefix + "unique_rate"] = len(cvals) / total
        vals[prefix + "singleton_share"] = sum(
            1 for c in cvals if c == 1) / total
        vals[prefix + "entropy"] = (
            _entropy_counts(cvals) if len(cvals) > 1 else 0.0)
        rp_den = total * (total - 1)
        vals[prefix + "repeat_pair_rate"] = (
            sum(c * (c - 1) for c in cvals) / rp_den if rp_den > 0 else 0.0)
    return vals


def _kept_names():
    return list(V3_CANDIDATE_NAMES if V3_KEPT is None else V3_KEPT)


def _build_coherent(chunk, names, out):
    kept = _kept_names()
    names.extend(kept)
    if out is None:
        return
    vals = _coherent_values(chunk)
    for name in kept:
        out.append(_safe_float(vals[name]))


def _feature_names():
    names = list(V2_FEATURE_NAMES)
    _build_coherent([], names, None)
    return names


FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)
COHERENT_FEATURE_NAMES = FEATURE_NAMES[N_FEATURES_V2:]


def extract_features(chunk):
    """Fixed-length v3 vector: v2 (148, bit-identical) + coherent block.

    Never raises: on any failure returns a zero vector of length N_FEATURES.
    """
    try:
        names, out = [], []
        _build_v1(chunk, names, out)
        _build_sig(chunk, names, out)
        _build_coherent(chunk, names, out)
        if len(out) != N_FEATURES:
            return [0.0] * N_FEATURES
        return [_safe_float(v) for v in out]
    except Exception:  # noqa: BLE001
        return [0.0] * N_FEATURES


if __name__ == "__main__":
    print(f"{N_FEATURES} features ({N_FEATURES_V1} v1 + "
          f"{N_FEATURES_V2 - N_FEATURES_V1} v2-sig + "
          f"{N_FEATURES - N_FEATURES_V2} coherent kept of "
          f"{len(V3_CANDIDATE_NAMES)} candidates)")
    for n in COHERENT_FEATURE_NAMES:
        print(" ", n)
