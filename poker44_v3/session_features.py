"""Native feature extraction for Poker44 v3.0 subject sessions.

Stdlib only (matching neurons/chunk_features*.py), never raises, and
vocabulary-agnostic: nothing here depends on knowing the concrete telemetry
``event_type`` strings, which are NOT enumerated anywhere in the subnet repo
(the only example in the whole tree is ``"pointer_click"``).  Every family
degrades to zeros when its inputs are missing, which is what makes this safe
to serve on day one of the format flip.

Design constraints taken from the contract
(contracts/subject-session.v1.schema.json on origin/dev):
  * ``hands[].hand_number`` is NOT reliably present -- the JSON schema
    requires it but the miner-side validator only checks that ``hands`` is a
    non-empty list, and the repo's own fixtures omit it.  Always ``.get()``.
  * telemetry ``events`` can be up to 50,000 per session and ``value`` is
    completely unconstrained JSON.  Both are handled defensively and the
    event scan is capped (``MAX_EVENTS_SCANNED``).
  * amounts are integers in unknown units.  Every size feature is a *ratio*,
    never an absolute, so the extractor is scale-free.

Feature vector is fixed-length and ordered; ``FEATURE_NAMES`` is the schema.
"""

import math
from collections import Counter

ACTION_TYPES = ("fold", "check", "call", "bet", "raise", "all_in")
PHASES = ("preflop", "flop", "turn", "river", "showdown")

# Hard cap on telemetry events examined per session.  A 256-session request at
# the schema's 50k-event ceiling would be 12.8M dicts; a pure-python pass over
# that busts even the 180s default timeout.  8k events is far more than enough
# for every statistic below to converge.
MAX_EVENTS_SCANNED = 8000
MAX_ACTIONS_SCANNED = 20000

# Substring probes for the telemetry vocabulary we cannot yet observe.  These
# are *hints*, not requirements: if none match, the corresponding features are
# 0.0 and the day-one scorer gives them zero variance, hence zero influence.
# After the first tournament window, replace these with the real vocabulary.
POINTER_TOKENS = ("pointer", "mouse", "move", "hover", "cursor", "drag")
CLICK_TOKENS = ("click", "tap", "press", "down", "up", "select")
KEY_TOKENS = ("key", "type", "input", "paste")
FOCUS_TOKENS = ("focus", "blur", "visib", "idle", "active", "scroll", "resize")


# --------------------------------------------------------------------------
# small numeric helpers (stdlib, never raise)
# --------------------------------------------------------------------------
def _f(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or out in (float("inf"), float("-inf")):
        return default
    return out


def _num_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _safe_div(num, den, default=0.0):
    den = _f(den)
    if den == 0.0:
        return default
    return _f(num) / den


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(math.floor(pos))
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _mad(values, median):
    if not values:
        return 0.0
    return _mean([abs(v - median) for v in values])


def _entropy(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p, 2)
    return ent


def _norm_entropy(counter):
    """Shannon entropy normalised by log2(k) -> [0, 1]; 0 when k < 2."""
    if not counter or len(counter) < 2:
        return 0.0
    return _entropy(list(counter.values())) / math.log(len(counter), 2)


def _round_share(values, modulus):
    if not values:
        return 0.0
    hits = 0
    for v in values:
        iv = int(round(v))
        if iv % modulus == 0:
            hits += 1
    return hits / len(values)


def _quantisation_score(values):
    """How close the sample is to being multiples of one grid step.

    Real humans produce ms timings with essentially no common divisor.  A bot
    that sleeps in fixed increments produces a large one.  Returns
    ``gcd / median`` clipped to [0, 1]; 0.0 when undefined.
    """
    ints = [int(round(v)) for v in values if v > 0]
    if len(ints) < 4:
        return 0.0
    g = 0
    for v in ints:
        g = math.gcd(g, v)
        if g == 1:
            return 0.0
    ordered = sorted(ints)
    med = _quantile(ordered, 0.5)
    if med <= 0:
        return 0.0
    return min(1.0, g / med)


def _lag1_autocorr(values):
    if len(values) < 4:
        return 0.0
    m = _mean(values)
    num = sum((values[i] - m) * (values[i + 1] - m) for i in range(len(values) - 1))
    den = sum((v - m) ** 2 for v in values)
    if den <= 0:
        return 0.0
    return max(-1.0, min(1.0, num / den))


def _burstiness(gaps):
    """Goh-Barabasi burstiness: +1 bursty (human), -1 perfectly periodic (bot)."""
    if len(gaps) < 3:
        return 0.0
    m = _mean(gaps)
    s = _std(gaps)
    if s + m <= 0:
        return 0.0
    return (s - m) / (s + m)


def _dist_stats(values, prefix, names, out):
    """Emit a fixed 20-slot distribution summary for a scalar sample."""
    ordered = sorted(values)
    n = len(ordered)
    mean = _mean(ordered)
    med = _quantile(ordered, 0.5)
    std = _std(ordered)
    counter = Counter(int(round(v)) for v in ordered)
    stats = [
        ("n_log", math.log1p(n)),
        ("mean_log", math.log1p(max(0.0, mean))),
        ("std_log", math.log1p(max(0.0, std))),
        ("cv", _safe_div(std, mean)),
        ("median_log", math.log1p(max(0.0, med))),
        ("mad_over_median", _safe_div(_mad(ordered, med), med)),
        ("iqr_over_median",
         _safe_div(_quantile(ordered, 0.75) - _quantile(ordered, 0.25), med)),
        ("p05_over_median", _safe_div(_quantile(ordered, 0.05), med)),
        ("p95_over_median", _safe_div(_quantile(ordered, 0.95), med)),
        ("tail_p99_over_p50", _safe_div(_quantile(ordered, 0.99), med)),
        ("min_over_median", _safe_div(ordered[0] if ordered else 0.0, med)),
        ("skew_proxy", _safe_div(mean - med, std)),
        ("distinct_ratio", _safe_div(len(counter), n)),
        ("mode_share", _safe_div(counter.most_common(1)[0][1] if counter else 0, n)),
        ("round10_share", _round_share(ordered, 10)),
        ("round100_share", _round_share(ordered, 100)),
        ("round1000_share", _round_share(ordered, 1000)),
        ("quantisation", _quantisation_score(ordered)),
        ("lag1_autocorr", _lag1_autocorr(values)),
        ("burstiness", _burstiness(values)),
    ]
    for suffix, value in stats:
        names.append(prefix + suffix)
        out.append(_f(value))


_DIST_SLOTS = 20


def _sequence_signature(tokens, prefix, names, out):
    """6 order/repetition statistics over a token stream (ported from v2/v3)."""
    n = len(tokens)
    counter = Counter(tokens)
    top1 = counter.most_common(1)[0][1] if counter else 0
    top2 = sum(c for _, c in counter.most_common(2))
    singles = sum(1 for c in counter.values() if c == 1)
    repeat_pairs = sum(1 for i in range(n - 1) if tokens[i] == tokens[i + 1])
    runs = 1 if n else 0
    for i in range(1, n):
        if tokens[i] != tokens[i - 1]:
            runs += 1
    stats = [
        ("top1_share", _safe_div(top1, n)),
        ("top2_share", _safe_div(top2, n)),
        ("unique_rate", _safe_div(len(counter), n)),
        ("singleton_share", _safe_div(singles, len(counter))),
        ("entropy", _norm_entropy(counter)),
        ("repeat_pair_rate", _safe_div(repeat_pairs, max(1, n - 1))),
        ("mean_run_len", _safe_div(n, runs)),
    ]
    for suffix, value in stats:
        names.append(prefix + suffix)
        out.append(_f(value))


_SIG_SLOTS = 7


def _token_share(counter, tokens):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    hits = 0
    for name, count in counter.items():
        low = str(name).lower()
        if any(tok in low for tok in tokens):
            hits += count
    return hits / total


def _strip_forbidden(value, dropped, path="session"):
    """Return a copy with label-ish keys removed anywhere in the tree.

    The reference ``MinerInferenceService`` RAISES when it finds these keys,
    which is a live footgun: telemetry ``value`` is unconstrained JSON and a
    click event carrying a button caption (``value={"label": "Raise to 200"}``)
    is a perfectly legal payload.  Raising there means a zero score for the
    whole window.  We strip and log instead, and never read the values.
    """
    forbidden = {"is_bot", "is_human", "ground_truth", "label", "bot_family"}
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            child_path = path + "." + str(key)
            if str(key).lower() in forbidden:
                dropped.append(child_path)
                continue
            out[key] = _strip_forbidden(child, dropped, child_path)
        return out
    if isinstance(value, list):
        return [
            _strip_forbidden(child, dropped, "%s[%d]" % (path, i))
            for i, child in enumerate(value)
        ]
    return value


# --------------------------------------------------------------------------
# main extractor
# --------------------------------------------------------------------------
def _collect_actions(session):
    hands = session.get("hands")
    if not isinstance(hands, list):
        return [], []
    per_hand = []
    flat = []
    for hand in hands:
        if not isinstance(hand, dict):
            per_hand.append([])
            continue
        raw = hand.get("actions")
        actions = [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []
        per_hand.append(actions)
        flat.extend(actions)
        if len(flat) >= MAX_ACTIONS_SCANNED:
            break
    return per_hand, flat[:MAX_ACTIONS_SCANNED]


def extract(session, cheap=False):
    """Return ``(names, values)`` for one subject session.  Never raises.

    ``cheap=True`` skips the telemetry-event scan (the only expensive part)
    and emits zeros for that family, so a slow request can still answer.
    """
    names = []
    out = []
    try:
        return _extract_inner(session, names, out, cheap)
    except Exception:  # noqa: BLE001 - a malformed session must never 500 us
        names, out = [], []
        _extract_inner({}, names, out, cheap=True)
        return names, out


def _extract_inner(session, names, out, cheap):
    if not isinstance(session, dict):
        session = {}
    per_hand, actions = _collect_actions(session)
    telemetry = session.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
    summary = telemetry.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    events_raw = telemetry.get("events")
    events = []
    if not cheap and isinstance(events_raw, list):
        for ev in events_raw[:MAX_EVENTS_SCANNED]:
            if isinstance(ev, dict):
                events.append(ev)

    n_hands = len(per_hand)
    n_actions = len(actions)

    # ---- A/B: action timing -------------------------------------------
    decision_ms = []
    gap_ms = []
    for a in actions:
        v = _num_or_none(a.get("decision_time_ms"))
        if v is not None and v >= 0:
            decision_ms.append(v)
        v = _num_or_none(a.get("time_since_last_action_ms"))
        if v is not None and v >= 0:
            gap_ms.append(v)
    _dist_stats(decision_ms, "dt_", names, out)
    _dist_stats(gap_ms, "gap_", names, out)

    # Threshold shares on decision times: humans have a reaction-time floor,
    # scripted agents do not; humans also have rare very long tank-thinks.
    for label, lo, hi in (
        ("sub150", 0.0, 150.0), ("sub300", 0.0, 300.0), ("sub500", 0.0, 500.0),
        ("1s_3s", 1000.0, 3000.0), ("over10s", 10000.0, float("inf")),
        ("over30s", 30000.0, float("inf")),
    ):
        names.append("dt_frac_" + label)
        out.append(_safe_div(sum(1 for v in decision_ms if lo <= v < hi), len(decision_ms)))
    names.append("gap_frac_over30s")
    out.append(_safe_div(sum(1 for v in gap_ms if v >= 30000.0), len(gap_ms)))
    names.append("gap_frac_over120s")
    out.append(_safe_div(sum(1 for v in gap_ms if v >= 120000.0), len(gap_ms)))

    # Drift: humans get slower as a session wears on; bots are stationary.
    half = len(decision_ms) // 2
    first_half = _mean(decision_ms[:half]) if half else 0.0
    second_half = _mean(decision_ms[half:]) if half else 0.0
    names.append("dt_drift_ratio")
    out.append(_safe_div(second_half, first_half, 1.0))
    names.append("dt_drift_log")
    out.append(math.log1p(max(0.0, second_half)) - math.log1p(max(0.0, first_half)))

    # ---- C: telemetry summary (declared by the platform) ---------------
    s_events = _f(summary.get("event_count"))
    s_actions = _f(summary.get("action_count"))
    s_duration = _f(summary.get("duration_ms"))
    s_decisions = _f(summary.get("decision_count"))
    s_dmean = _f(summary.get("decision_mean_ms"))
    s_dstd = _f(summary.get("decision_std_ms"))
    summary_stats = [
        ("sum_event_count_log", math.log1p(max(0.0, s_events))),
        ("sum_action_count_log", math.log1p(max(0.0, s_actions))),
        ("sum_duration_log", math.log1p(max(0.0, s_duration))),
        ("sum_decision_count_log", math.log1p(max(0.0, s_decisions))),
        ("sum_decision_mean_log", math.log1p(max(0.0, s_dmean))),
        ("sum_decision_std_log", math.log1p(max(0.0, s_dstd))),
        ("sum_decision_cv", _safe_div(s_dstd, s_dmean)),
        ("sum_events_per_action", _safe_div(s_events, s_actions)),
        ("sum_events_per_second", _safe_div(s_events, s_duration / 1000.0)),
        ("sum_actions_per_minute", _safe_div(s_actions, s_duration / 60000.0)),
        ("sum_duty_cycle", _safe_div(s_decisions * s_dmean, s_duration)),
        ("sum_seconds_per_action", _safe_div(s_duration / 1000.0, s_actions)),
        # Cross-checks between the declared summary and what we can count.
        ("sum_action_mismatch", _safe_div(abs(s_actions - n_actions), max(1.0, s_actions))),
        ("sum_decision_mismatch",
         _safe_div(abs(s_decisions - len(decision_ms)), max(1.0, s_decisions))),
        ("sum_mean_mismatch",
         _safe_div(abs(s_dmean - _mean(decision_ms)), max(1.0, s_dmean))),
        ("sum_events_per_hand", _safe_div(s_events, n_hands)),
    ]
    for name, value in summary_stats:
        names.append(name)
        out.append(_f(value))

    # ---- D: telemetry events (vocabulary agnostic) ---------------------
    n_events = len(events)
    client = [e for e in events if str(e.get("source")) == "client"]
    server = [e for e in events if str(e.get("source")) == "server"]
    type_counter = Counter(str(e.get("event_type") or "") for e in events)
    client_type_counter = Counter(str(e.get("event_type") or "") for e in client)
    target_counter = Counter(
        str(e.get("target")) for e in events if e.get("target") is not None
    )
    offsets = []
    for e in client:
        v = _num_or_none(e.get("offset_ms"))
        if v is not None and v >= 0:
            offsets.append(v)
    offsets.sort()
    ev_gaps = [offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1)]

    seqs = [_num_or_none(e.get("sequence")) for e in events]
    seqs = [s for s in seqs if s is not None]
    monotone = sum(1 for i in range(len(seqs) - 1) if seqs[i + 1] > seqs[i])
    value_present = sum(1 for e in events if e.get("value") is not None)
    value_dict = sum(1 for e in events if isinstance(e.get("value"), dict))
    value_num = sum(
        1 for e in events
        if isinstance(e.get("value"), (int, float)) and not isinstance(e.get("value"), bool)
    )

    event_stats = [
        ("ev_n_log", math.log1p(n_events)),
        ("ev_client_frac", _safe_div(len(client), n_events)),
        ("ev_server_frac", _safe_div(len(server), n_events)),
        ("ev_distinct_types_log", math.log1p(len(type_counter))),
        ("ev_type_entropy", _norm_entropy(type_counter)),
        ("ev_type_top1_share",
         _safe_div(type_counter.most_common(1)[0][1] if type_counter else 0, n_events)),
        ("ev_client_distinct_types_log", math.log1p(len(client_type_counter))),
        ("ev_distinct_targets_log", math.log1p(len(target_counter))),
        ("ev_target_entropy", _norm_entropy(target_counter)),
        ("ev_null_target_frac",
         _safe_div(sum(1 for e in events if e.get("target") is None), n_events)),
        ("ev_targets_per_type", _safe_div(len(target_counter), max(1, len(type_counter)))),
        ("ev_value_present_frac", _safe_div(value_present, n_events)),
        ("ev_value_dict_frac", _safe_div(value_dict, n_events)),
        ("ev_value_numeric_frac", _safe_div(value_num, n_events)),
        ("ev_per_action", _safe_div(n_events, n_actions)),
        ("ev_client_per_action", _safe_div(len(client), n_actions)),
        ("ev_server_per_action", _safe_div(len(server), n_actions)),
        ("ev_per_hand", _safe_div(n_events, n_hands)),
        ("ev_per_second", _safe_div(n_events, s_duration / 1000.0)),
        ("ev_span_coverage", _safe_div(offsets[-1] - offsets[0] if len(offsets) > 1 else 0.0,
                                       s_duration)),
        ("ev_seq_monotone_frac", _safe_div(monotone, max(1, len(seqs) - 1))),
        ("ev_seq_distinct_ratio", _safe_div(len(set(seqs)), len(seqs))),
        # Vocabulary probes -- 0.0 until the real event names are observed.
        ("ev_pointer_share", _token_share(type_counter, POINTER_TOKENS)),
        ("ev_click_share", _token_share(type_counter, CLICK_TOKENS)),
        ("ev_key_share", _token_share(type_counter, KEY_TOKENS)),
        ("ev_focus_share", _token_share(type_counter, FOCUS_TOKENS)),
    ]
    for name, value in event_stats:
        names.append(name)
        out.append(_f(value))
    _dist_stats(ev_gaps, "evgap_", names, out)

    # ---- E: poker behaviour (survives the sanitisation) ----------------
    a_types = [str(a.get("action_type") or "") for a in actions]
    a_phases = [str(a.get("phase") or "") for a in actions]
    type_counts = Counter(a_types)
    phase_counts = Counter(a_phases)
    aggressive = type_counts["bet"] + type_counts["raise"] + type_counts["all_in"]
    passive = type_counts["check"] + type_counts["call"]

    for t in ACTION_TYPES:
        names.append("act_share_" + t)
        out.append(_safe_div(type_counts[t], n_actions))
    names.append("act_aggression_freq")
    out.append(_safe_div(aggressive, n_actions))
    names.append("act_aggr_to_passive")
    out.append(_safe_div(aggressive, passive))
    names.append("act_fold_rate")
    out.append(_safe_div(type_counts["fold"], n_actions))
    names.append("act_allin_share")
    out.append(_safe_div(type_counts["all_in"], n_actions))
    names.append("act_is_allin_flag_share")
    out.append(_safe_div(sum(1 for a in actions if a.get("is_all_in") is True), n_actions))

    for p in PHASES:
        in_phase = [a_types[i] for i in range(n_actions) if a_phases[i] == p]
        c = Counter(in_phase)
        names.append("phase_share_" + p)
        out.append(_safe_div(phase_counts[p], n_actions))
        names.append("phase_aggr_" + p)
        out.append(_safe_div(c["bet"] + c["raise"] + c["all_in"], len(in_phase)))
        names.append("phase_fold_" + p)
        out.append(_safe_div(c["fold"], len(in_phase)))
        names.append("phase_reach_" + p)
        out.append(_safe_div(
            sum(1 for h in per_hand if any(str(a.get("phase") or "") == p for a in h)),
            n_hands))

    names.append("hands_n_log")
    out.append(math.log1p(n_hands))
    names.append("actions_n_log")
    out.append(math.log1p(n_actions))
    per_hand_counts = [float(len(h)) for h in per_hand]
    names.append("actions_per_hand_mean")
    out.append(_mean(per_hand_counts))
    names.append("actions_per_hand_std")
    out.append(_std(per_hand_counts))
    names.append("actions_per_hand_cv")
    out.append(_safe_div(_std(per_hand_counts), _mean(per_hand_counts)))
    names.append("distinct_phases")
    out.append(float(len([p for p in phase_counts if p])))

    _sequence_signature(a_types, "sig_act_", names, out)
    _sequence_signature(a_phases, "sig_phase_", names, out)
    _sequence_signature(
        [a_phases[i] + "|" + a_types[i] for i in range(n_actions)],
        "sig_pa_", names, out)
    _sequence_signature(
        [str(a.get("event_type") or "") for a in actions], "sig_evt_", names, out)
    _sequence_signature(
        [str(a.get("position_name") or "") for a in actions], "sig_pos_", names, out)

    # ---- F: bet sizing (solver bots snap to canonical pot fractions) ----
    pot_fracs = []
    stack_fracs = []
    call_ratios = []
    raise_values = []
    for a in actions:
        amount = _num_or_none(a.get("amount"))
        pot = _num_or_none(a.get("pot_size"))
        stack = _num_or_none(a.get("player_stack"))
        call_amt = _num_or_none(a.get("call_amount"))
        raise_to = _num_or_none(a.get("raise_to"))
        if amount is not None and amount > 0 and pot is not None and pot > 0:
            pot_fracs.append(amount / pot)
        if amount is not None and amount > 0 and stack is not None and stack > 0:
            stack_fracs.append(amount / stack)
        if call_amt is not None and call_amt > 0 and pot is not None and pot > 0:
            call_ratios.append(call_amt / pot)
        if raise_to is not None and raise_to > 0:
            raise_values.append(raise_to)

    names.append("bet_pot_frac_mean")
    out.append(_mean(pot_fracs))
    names.append("bet_pot_frac_std")
    out.append(_std(pot_fracs))
    names.append("bet_pot_frac_cv")
    out.append(_safe_div(_std(pot_fracs), _mean(pot_fracs)))
    names.append("bet_pot_frac_distinct_ratio")
    out.append(_safe_div(len({round(v, 3) for v in pot_fracs}), len(pot_fracs)))
    snapped = 0
    for v in pot_fracs:
        for target in (0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5, 2.0):
            if abs(v - target) <= 0.02:
                snapped += 1
                break
    names.append("bet_pot_frac_snap_share")
    out.append(_safe_div(snapped, len(pot_fracs)))
    names.append("bet_stack_frac_mean")
    out.append(_mean(stack_fracs))
    names.append("bet_call_ratio_mean")
    out.append(_mean(call_ratios))
    names.append("bet_raise_round10_share")
    out.append(_round_share(raise_values, 10))
    names.append("bet_raise_round100_share")
    out.append(_round_share(raise_values, 100))
    names.append("bet_raise_distinct_ratio")
    out.append(_safe_div(len({int(round(v)) for v in raise_values}), len(raise_values)))

    # ---- G: structure / table context ----------------------------------
    active = [v for v in (_num_or_none(a.get("active_players")) for a in actions)
              if v is not None]
    seats = [v for v in (_num_or_none(a.get("seat_position")) for a in actions)
             if v is not None]
    names.append("tab_active_mean")
    out.append(_mean(active))
    names.append("tab_active_std")
    out.append(_std(active))
    names.append("tab_distinct_seats")
    out.append(float(len(set(int(s) for s in seats))))
    names.append("tab_hole_cards_frac")
    out.append(_safe_div(sum(1 for a in actions if a.get("hole_cards")), n_actions))
    names.append("tab_community_cards_frac")
    out.append(_safe_div(sum(1 for a in actions if a.get("community_cards")), n_actions))
    names.append("tab_occurred_at_frac")
    out.append(_safe_div(sum(1 for a in actions if a.get("occurred_at")), n_actions))
    a_seqs = [v for v in (_num_or_none(a.get("sequence")) for a in actions) if v is not None]
    names.append("tab_action_seq_monotone")
    out.append(_safe_div(
        sum(1 for i in range(len(a_seqs) - 1) if a_seqs[i + 1] > a_seqs[i]),
        max(1, len(a_seqs) - 1)))
    names.append("tab_cheap_mode")
    out.append(1.0 if cheap else 0.0)

    return names, out


def feature_names():
    """Canonical ordered feature names (from an empty session)."""
    names, _ = extract({}, cheap=False)
    return names


FEATURE_NAMES = feature_names()
N_FEATURES = len(FEATURE_NAMES)


def telemetry_vocabulary(sessions):
    """Distil the observed telemetry vocabulary from a batch of sessions.

    This is the intel product of the format reset: the concrete
    ``event_type`` / ``target`` / ``value``-shape vocabulary is documented
    NOWHERE in the subnet repo, so the first tournament window is the only
    place to learn it.  Called on every request by the capture hook.
    """
    ev_types = Counter()
    ev_targets = Counter()
    ev_sources = Counter()
    value_shapes = Counter()
    value_samples = {}
    act_event_types = Counter()
    positions = Counter()
    phases = Counter()
    for session in sessions if isinstance(sessions, list) else []:
        if not isinstance(session, dict):
            continue
        telemetry = session.get("telemetry")
        if isinstance(telemetry, dict):
            events = telemetry.get("events")
            if isinstance(events, list):
                for ev in events[:MAX_EVENTS_SCANNED]:
                    if not isinstance(ev, dict):
                        continue
                    et = str(ev.get("event_type") or "")
                    ev_types[et] += 1
                    ev_sources[str(ev.get("source"))] += 1
                    if ev.get("target") is not None:
                        ev_targets[str(ev.get("target"))] += 1
                    val = ev.get("value")
                    shape = type(val).__name__
                    if isinstance(val, dict):
                        shape = "dict{" + ",".join(sorted(str(k) for k in val)[:8]) + "}"
                    value_shapes[et + " -> " + shape] += 1
                    key = et + "|" + shape
                    if key not in value_samples and val is not None:
                        try:
                            value_samples[key] = repr(val)[:200]
                        except Exception:  # noqa: BLE001
                            pass
        hands = session.get("hands")
        if isinstance(hands, list):
            for hand in hands:
                if not isinstance(hand, dict):
                    continue
                acts = hand.get("actions")
                if not isinstance(acts, list):
                    continue
                for a in acts:
                    if isinstance(a, dict):
                        act_event_types[str(a.get("event_type") or "")] += 1
                        positions[str(a.get("position_name"))] += 1
                        phases[str(a.get("phase"))] += 1
    return {
        "telemetry_event_types": ev_types.most_common(200),
        "telemetry_targets": ev_targets.most_common(200),
        "telemetry_sources": ev_sources.most_common(),
        "telemetry_value_shapes": value_shapes.most_common(200),
        "telemetry_value_samples": value_samples,
        "hand_action_event_types": act_event_types.most_common(100),
        "position_names": positions.most_common(50),
        "phases": phases.most_common(20),
    }
