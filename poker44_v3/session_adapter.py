"""Adapter: v3.0 subject session -> legacy chunk shape.

PURPOSE AND HONEST LIMITS
-------------------------
This lets ``neurons/chunk_features_v3.extract_features`` (our deployed
468-feature extractor) run against v3.0 payloads.  That is useful for two
things:

  1. OFFLINE research and post-tournament training -- reusing 468 columns of
     already-validated behavioural engineering instead of starting over.
  2. An OPTIONAL blend inside the live scorer, off by default.

It is NOT a drop-in rescue for the deployed GBM, and the staged miner does
NOT use it in the serving path by default (``POKER44_V3_CHUNK_BLEND=0.0``).
Reason: v3.0 deleted the fields a large share of those 468 features are built
on --

    GONE ENTIRELY : metadata{game_type,limit_type,max_seats,hero_seat,
                    button_seat,sb,bb,ante,hand_ended_on_street,
                    rng_seed_commitment}, the whole players[] roster
                    (starting_stack, showed_hand), streets[]/board_cards,
                    and the entire outcome{} block (winners, payouts,
                    total_pot, rake, showdown, result_reason).
    RENAMED       : street->phase, actor_seat->seat_position,
                    call_to->call_amount, pot_before/pot_after->
                    pot_size/current_bet.
    NEW           : sequence, event_type, is_all_in, active_players,
                    position_name, community_cards, populated hole_cards,
                    decision_time_ms, time_since_last_action_ms, occurred_at,
                    plus the whole telemetry block.

Everything in the GONE list is reconstructed here as a best-effort estimate or
left absent, so those features become constant across the batch.  A gradient
booster fed constant columns it was trained to treat as informative produces
predictions that are stable but not meaningfully ranked -- which under a
rank-based reward is close to worthless, and could be worse than neutral.
Treat the adapted score as a research signal until it is validated against
real labels.

The reconstructions are documented inline so nobody later mistakes an
estimate for ground truth.
"""

from collections import Counter


def _num(value, default=None):
    if value is None or isinstance(value, bool):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out or abs(out) == float("inf"):
        return default
    return out


def _infer_big_blind(actions):
    """ESTIMATE.  The real ``bb`` is gone from the payload.

    Heuristic: the modal smallest positive preflop ``call_amount`` is, in most
    ring games, one big blind.  Falls back to the smallest positive amount,
    then to 1.0.  Only ratios computed from this matter, so a constant
    mis-scale across a session is harmless; a per-session mis-scale is not,
    which is another reason this path stays off by default.
    """
    preflop_calls = [
        _num(a.get("call_amount")) for a in actions
        if str(a.get("phase") or "").lower() == "preflop"
    ]
    preflop_calls = [v for v in preflop_calls if v and v > 0]
    if preflop_calls:
        return Counter(preflop_calls).most_common(1)[0][0]
    amounts = [_num(a.get("amount")) for a in actions]
    amounts = [v for v in amounts if v and v > 0]
    if amounts:
        return min(amounts)
    return 1.0


def session_to_chunk(session):
    """Return a legacy-shaped ``chunk`` (list of hand dicts) for one session."""
    if not isinstance(session, dict):
        return []
    hands = session.get("hands")
    if not isinstance(hands, list):
        return []

    flat = [
        a for hand in hands if isinstance(hand, dict)
        for a in (hand.get("actions") or []) if isinstance(a, dict)
    ]
    big_blind = _infer_big_blind(flat)

    # ESTIMATE: the subject's seat.  v3.0 does not label a hero, but a subject
    # session is by construction one player's view, so the modal seat_position
    # (or the seat that holds hole_cards) is the subject.
    seat_with_cards = [
        a.get("seat_position") for a in flat
        if a.get("hole_cards") and a.get("seat_position") is not None
    ]
    seats = [a.get("seat_position") for a in flat if a.get("seat_position") is not None]
    if seat_with_cards:
        hero_seat = Counter(seat_with_cards).most_common(1)[0][0]
    elif seats:
        hero_seat = Counter(seats).most_common(1)[0][0]
    else:
        hero_seat = 0
    max_seats = 0
    for a in flat:
        ap = _num(a.get("active_players"), 0) or 0
        if ap > max_seats:
            max_seats = ap
    if not max_seats:
        max_seats = len({s for s in seats}) or 6

    out = []
    for index, hand in enumerate(hands):
        if not isinstance(hand, dict):
            continue
        raw = hand.get("actions")
        actions = [a for a in raw if isinstance(a, dict)] if isinstance(raw, list) else []

        legacy_actions = []
        stacks = {}
        boards = {}
        for a in actions:
            amount = _num(a.get("amount"), 0.0) or 0.0
            pot_before = _num(a.get("pot_size"), 0.0) or 0.0
            seat = a.get("seat_position")
            stack = _num(a.get("player_stack"))
            if seat is not None and stack is not None and seat not in stacks:
                # ESTIMATE: starting_stack is gone; the first observed stack
                # for a seat is the closest available proxy.
                stacks[seat] = stack
            phase = a.get("phase")
            if phase and a.get("community_cards"):
                boards.setdefault(str(phase), list(a.get("community_cards") or []))
            legacy_actions.append({
                "action_id": str(a.get("sequence", len(legacy_actions))),
                "street": a.get("phase"),
                "actor_seat": seat,
                "action_type": a.get("action_type"),
                "amount": amount,
                "raise_to": _num(a.get("raise_to")),
                "call_to": _num(a.get("call_amount")),
                "normalized_amount_bb": (amount / big_blind) if big_blind else 0.0,
                "pot_before": pot_before,
                # ESTIMATE: pot_after is gone; pot_size + committed amount is
                # the natural reconstruction for the actor's own commitment.
                "pot_after": pot_before + amount,
            })

        out.append({
            "metadata": {
                "game_type": "Hold'em",           # ESTIMATE: not in v3.0
                "limit_type": "No Limit",         # ESTIMATE: not in v3.0
                "max_seats": int(max_seats),
                "hero_seat": hero_seat,
                "hand_ended_on_street": "",       # GONE
                "button_seat": 0,                 # GONE (main sanitiser also forced 0)
                "sb": big_blind / 2.0,            # ESTIMATE
                "bb": big_blind,                  # ESTIMATE
                "ante": 0.0,                      # GONE
                "rng_seed_commitment": None,      # GONE
            },
            "players": [
                {"player_uid": "seat_%s" % seat, "seat": seat,
                 "starting_stack": stacks[seat], "hole_cards": None,
                 "showed_hand": False}
                for seat in sorted(stacks, key=lambda s: (s is None, s))
            ],
            "streets": [
                {"street": street, "board_cards": cards}
                for street, cards in boards.items()
            ],
            "actions": legacy_actions,
            "outcome": {},                        # GONE entirely in v3.0
            "hand_number": hand.get("hand_number", index + 1),
        })
    return out


def sessions_to_chunks(sessions):
    """One legacy chunk per session, preserving order."""
    if not isinstance(sessions, list):
        return []
    return [session_to_chunk(s) for s in sessions]
