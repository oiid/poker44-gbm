"""Chunk-level feature extraction for Poker44 bot detection.

A chunk is a list of sanitized hand dicts (metadata/players/streets/actions/
outcome).  ``metadata.hero_seat`` marks the focus player whose behavior the
chunk represents.  Amounts are noisy bucketed big-blind values and only a
sample of actions is visible per hand, so features are aggregates that are
robust to missing data.

Exposes:
    FEATURE_NAMES: list[str]
    extract_features(chunk: list[dict]) -> list[float]

Stdlib-only; never raises on malformed input (returns zeros instead).
"""

from __future__ import annotations

import math

STREETS = ("preflop", "flop", "turn", "river")
ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
AGGRESSIVE = ("bet", "raise")


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _quantile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _dist_stats(xs, prefix, names, out, n_hands=1):
    """mean/std/cv/min/q25/median/q75/max + distinct-count features.

    Distinct counts are reported per hand (``n_distinct_per_hand``) rather
    than raw so the feature does not scale with the number of hands in the
    chunk (live groups can be 2-3x the benchmark sub-chunk size).
    """
    names.extend([
        f"{prefix}_mean", f"{prefix}_std", f"{prefix}_cv",
        f"{prefix}_min", f"{prefix}_q25", f"{prefix}_median",
        f"{prefix}_q75", f"{prefix}_max",
        f"{prefix}_n_distinct_per_hand", f"{prefix}_distinct_ratio",
    ])
    if out is None:
        return
    m = _mean(xs)
    sd = _std(xs)
    cv = sd / abs(m) if abs(m) > 1e-9 else 0.0
    distinct = len({round(x, 4) for x in xs})
    out.extend([
        m, sd, cv,
        min(xs) if xs else 0.0,
        _quantile(xs, 0.25), _quantile(xs, 0.5), _quantile(xs, 0.75),
        max(xs) if xs else 0.0,
        distinct / max(n_hands, 1),
        distinct / len(xs) if xs else 0.0,
    ])


def _build(chunk, names, out):
    """Single pass that both registers feature names and (optionally) values.

    When ``out`` is None only ``names`` is filled, using an empty chunk.
    """
    hands = chunk if isinstance(chunk, (list, tuple)) else []

    # ---- per-hand accumulators -------------------------------------------
    n_hands = 0
    hero_type_counts = {t: 0 for t in ACTION_TYPES}
    hero_total = 0
    all_type_counts = {t: 0 for t in ACTION_TYPES}
    all_total = 0
    hero_street_totals = {s: 0 for s in STREETS}
    hero_street_aggr = {s: 0 for s in STREETS}
    hero_street_fold = {s: 0 for s in STREETS}
    street_reach = {s: 0 for s in STREETS}  # hands with any action on street

    hero_amounts = []          # nonzero normalized_amount_bb of hero actions
    hero_raise_to = []
    hero_call_to = []
    hero_bet_ratio = []        # hero aggressive amount / pot_before (bb)
    pot_before_bb = []
    pot_after_bb = []
    pot_growth = []

    vis_action_counts = []
    hero_action_counts = []
    hero_per_hand_aggr = []    # per-hand hero aggression frequency
    hero_per_hand_amt_mean = []
    hero_stacks_bb = []
    n_players_list = []
    bb_values = []
    hero_first_fold_preflop = 0
    hero_any_action_hands = 0
    hero_vpip_hands = 0        # hero call/bet/raise at least once in hand

    for h in hands:
        if not isinstance(h, dict):
            continue
        n_hands += 1
        meta = h.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        hero_seat = meta.get("hero_seat")
        bb = _safe_float(meta.get("bb"), 0.0)
        if bb > 0:
            bb_values.append(bb)
        players = h.get("players") or []
        if isinstance(players, list):
            n_players_list.append(float(len(players)))
            for p in players:
                if isinstance(p, dict) and p.get("seat") == hero_seat:
                    st = _safe_float(p.get("starting_stack"))
                    if bb > 0 and st > 0:
                        hero_stacks_bb.append(st / bb)

        actions = h.get("actions") or []
        if not isinstance(actions, list):
            actions = []
        vis_action_counts.append(float(len(actions)))

        streets_hit = set()
        h_hero_total = 0
        h_hero_aggr = 0
        h_hero_amts = []
        hero_voluntary = False
        hero_folded_preflop = False

        for a in actions:
            if not isinstance(a, dict):
                continue
            at = a.get("action_type")
            st = a.get("street")
            if st in street_reach:
                streets_hit.add(st)
            if at in all_type_counts:
                all_type_counts[at] += 1
                all_total += 1
            pb = _safe_float(a.get("pot_before"))
            pa = _safe_float(a.get("pot_after"))
            if bb > 0 and pb > 0:
                pot_before_bb.append(pb / bb)
            if bb > 0 and pa > 0:
                pot_after_bb.append(pa / bb)
            if pb > 1e-9 and pa > 0:
                pot_growth.append(pa / pb)

            if hero_seat is None or a.get("actor_seat") != hero_seat:
                continue
            # ---- hero action ----
            h_hero_total += 1
            if at in hero_type_counts:
                hero_type_counts[at] += 1
                hero_total += 1
            if st in hero_street_totals and at in ACTION_TYPES:
                hero_street_totals[st] += 1
                if at in AGGRESSIVE:
                    hero_street_aggr[st] += 1
                if at == "fold":
                    hero_street_fold[st] += 1
            if at in AGGRESSIVE:
                h_hero_aggr += 1
            if at in ("call", "bet", "raise"):
                hero_voluntary = True
            if at == "fold" and st == "preflop":
                hero_folded_preflop = True
            amt = _safe_float(a.get("normalized_amount_bb"))
            if amt > 0:
                hero_amounts.append(amt)
                h_hero_amts.append(amt)
            rt = _safe_float(a.get("raise_to"))
            if a.get("raise_to") is not None and rt > 0 and bb > 0:
                hero_raise_to.append(rt / bb)
            ct = _safe_float(a.get("call_to"))
            if a.get("call_to") is not None and ct > 0 and bb > 0:
                hero_call_to.append(ct / bb)
            if at in AGGRESSIVE and amt > 0 and bb > 0 and pb > 1e-9:
                hero_bet_ratio.append(amt / (pb / bb))

        for s in streets_hit:
            street_reach[s] += 1
        hero_action_counts.append(float(h_hero_total))
        if h_hero_total > 0:
            hero_any_action_hands += 1
            hero_per_hand_aggr.append(h_hero_aggr / h_hero_total)
            if hero_voluntary:
                hero_vpip_hands += 1
            if hero_folded_preflop and not hero_voluntary:
                hero_first_fold_preflop += 1
        if h_hero_amts:
            hero_per_hand_amt_mean.append(_mean(h_hero_amts))

    nh = max(n_hands, 1)
    nah = max(hero_any_action_hands, 1)

    def emit(name, value):
        names.append(name)
        if out is not None:
            out.append(_safe_float(value))

    # ---- global counts ----------------------------------------------------
    # n_hands is kept raw on purpose: it tells the model the group size so it
    # can calibrate; labels are balanced across sizes in training so it
    # carries no label signal by itself.
    emit("n_hands", float(n_hands))
    # Per-hand rate instead of a raw total so the feature does not scale
    # with group size (live groups are ~80-105 hands vs 30-40 in training).
    emit("hero_actions_per_hand", hero_total / nh)
    emit("hero_actions_per_hand_mean", _mean(hero_action_counts))
    emit("hero_actions_per_hand_std", _std(hero_action_counts))
    emit("vis_actions_per_hand_mean", _mean(vis_action_counts))
    emit("vis_actions_per_hand_std", _std(vis_action_counts))
    emit("vis_actions_per_hand_min",
         min(vis_action_counts) if vis_action_counts else 0.0)
    emit("vis_actions_per_hand_max",
         max(vis_action_counts) if vis_action_counts else 0.0)

    # ---- hero action-type shares -----------------------------------------
    ht = max(hero_total, 1)
    for t in ACTION_TYPES:
        emit(f"hero_share_{t}", hero_type_counts[t] / ht)
    emit("hero_aggression_freq",
         (hero_type_counts["bet"] + hero_type_counts["raise"]) / ht)
    calls_checks = hero_type_counts["call"] + hero_type_counts["check"]
    emit("hero_aggr_to_passive",
         (hero_type_counts["bet"] + hero_type_counts["raise"])
         / max(calls_checks, 1))

    # ---- table (all players) action-type shares ---------------------------
    att = max(all_total, 1)
    for t in ACTION_TYPES:
        emit(f"table_share_{t}", all_type_counts[t] / att)
    emit("hero_action_fraction_of_table", hero_total / att)

    # ---- hero per-street behavior -----------------------------------------
    for s in STREETS:
        emit(f"hero_{s}_action_share", hero_street_totals[s] / ht)
    for s in STREETS:
        emit(f"hero_{s}_aggr_freq",
             hero_street_aggr[s] / max(hero_street_totals[s], 1))
    for s in STREETS:
        emit(f"hero_{s}_fold_rate",
             hero_street_fold[s] / max(hero_street_totals[s], 1))

    # ---- street reach rates ------------------------------------------------
    for s in STREETS:
        emit(f"reach_{s}_rate", street_reach[s] / nh)
    emit("reach_turn_given_flop",
         street_reach["turn"] / max(street_reach["flop"], 1))
    emit("reach_river_given_turn",
         street_reach["river"] / max(street_reach["turn"], 1))

    # ---- per-hand hero tendencies ------------------------------------------
    emit("hero_vpip_rate", hero_vpip_hands / nah)
    emit("hero_fold_preflop_only_rate", hero_first_fold_preflop / nah)
    emit("hero_per_hand_aggr_mean", _mean(hero_per_hand_aggr))
    emit("hero_per_hand_aggr_std", _std(hero_per_hand_aggr))
    emit("hero_per_hand_amt_mean_std", _std(hero_per_hand_amt_mean))

    # ---- sizing distributions ----------------------------------------------
    _dist_stats(hero_amounts, "hero_amt_bb", names, out, n_hands=nh)
    _dist_stats(hero_raise_to, "hero_raise_to_bb", names, out, n_hands=nh)
    _dist_stats(hero_call_to, "hero_call_to_bb", names, out, n_hands=nh)
    _dist_stats(hero_bet_ratio, "hero_bet_to_pot", names, out, n_hands=nh)
    _dist_stats(pot_before_bb, "pot_before_bb", names, out, n_hands=nh)
    _dist_stats(pot_after_bb, "pot_after_bb", names, out, n_hands=nh)
    emit("pot_growth_mean", _mean(pot_growth))
    emit("pot_growth_std", _std(pot_growth))

    # ---- stacks / table -----------------------------------------------------
    emit("hero_stack_bb_mean", _mean(hero_stacks_bb))
    emit("hero_stack_bb_std", _std(hero_stacks_bb))
    emit("n_players_mean", _mean(n_players_list))
    emit("bb_mean", _mean(bb_values))
    emit("bb_n_distinct", float(len({round(b, 6) for b in bb_values})))


def _feature_names():
    names = []
    _build([], names, None)
    return names


FEATURE_NAMES = _feature_names()
N_FEATURES = len(FEATURE_NAMES)


def extract_features(chunk):
    """Extract a fixed-length feature vector from a chunk of hands.

    Never raises: on any failure returns a zero vector of length N_FEATURES.
    """
    try:
        names, out = [], []
        _build(chunk, names, out)
        if len(out) != N_FEATURES:
            return [0.0] * N_FEATURES
        return [_safe_float(v) for v in out]
    except Exception:
        return [0.0] * N_FEATURES


if __name__ == "__main__":
    print(f"{N_FEATURES} features")
    for n in FEATURE_NAMES:
        print(" ", n)
