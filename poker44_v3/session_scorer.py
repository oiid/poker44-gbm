"""Day-one v3.0 scorer: one calibrated bot-risk score per subject session.

THE PROBLEM THIS SOLVES
-----------------------
When v3.0 flips we will have ZERO labelled telemetry.  Our 468-feature chunk
ensemble is trained on fields that no longer exist in the payload (metadata,
players roster, streets/board, the whole outcome/showdown block).  Serving it
through an adapter would be an out-of-distribution guess.  Serving nothing at
all is a guaranteed 0.0 reward.

So day one runs a *prior-driven, in-batch rank ensemble*: a set of
hand-crafted sub-signals each with a known sign (higher => more bot),
rank-normalised across the sessions in the request, then blended.

WHY IN-BATCH RANKS ARE THE RIGHT CALL WITH NO TRAINING DATA
-----------------------------------------------------------
1. The validator sends the ENTIRE evaluation window in one synapse
   (``SessionDetectionSynapse(sessions=validation_round.miner_sessions)``),
   and computes AP / recall@FPR over exactly that batch.  So the batch we see
   IS the scoring population -- ranking within it is exactly what the reward
   measures.
2. Reward = 0.50*AP_skill + 0.30*recall@FPR5% + 0.20*brier_skill.  Two of the
   three terms are pure ranking; only the last needs calibration.
3. We do not know the units, scale or vocabulary of the new payload.  Ranks
   are invariant to all of that.
4. A sub-signal whose inputs are missing has zero variance across the batch,
   so it collapses to a constant 0.5 and contributes NOTHING.  The ensemble
   degrades automatically instead of degrading wrongly.

FALLBACK CHAIN (never returns fewer/other-shaped scores, never raises)
   trained artifact (if POKER44_SESSION_MODEL_ARTIFACT is set and loads)
     -> rank ensemble (default; needs >= MIN_BATCH_FOR_RANKS sessions)
     -> absolute-threshold heuristic (tiny batches / single session)
     -> constant 0.5 (a malformed batch still gets a valid, finite response)
"""

import math
import os
import time

from poker44_v3 import session_features as sf

# --------------------------------------------------------------------------
# Sub-signal table: (feature_name, sign, weight)
#   sign=+1 : larger feature value => more bot-like
#   sign=-1 : larger feature value => more human-like
# Weights are priors, not fitted numbers.  They are deliberately spread over
# independent families so no single unverified assumption can dominate.
# --------------------------------------------------------------------------
SIGNALS = (
    # --- decision-timing regularity: the single strongest a-priori tell ---
    ("dt_cv", -1.0, 1.20),                  # humans vary a lot, scripts do not
    ("sum_decision_cv", -1.0, 1.20),        # same, from the platform's summary
    ("dt_mad_over_median", -1.0, 0.80),
    ("dt_iqr_over_median", -1.0, 0.80),
    ("dt_distinct_ratio", -1.0, 0.90),      # bots repeat exact ms values
    ("dt_mode_share", +1.0, 0.70),
    ("dt_quantisation", +1.0, 1.00),        # sleep(k) leaves a common divisor
    ("dt_round100_share", +1.0, 0.60),
    ("dt_round1000_share", +1.0, 0.50),
    ("dt_burstiness", -1.0, 0.70),          # human action streams are bursty
    ("dt_tail_p99_over_p50", -1.0, 0.60),   # humans tank occasionally
    ("dt_frac_sub300", +1.0, 0.70),         # below human reaction floor
    ("dt_frac_sub150", +1.0, 0.60),
    ("dt_frac_over10s", -1.0, 0.50),
    ("dt_mean_log", -1.0, 0.40),
    ("dt_drift_log", -1.0, 0.50),           # humans fatigue and slow down

    # --- idle-gap structure: humans have heavy-tailed away-from-keyboard ---
    ("gap_tail_p99_over_p50", -1.0, 0.60),
    ("gap_frac_over30s", -1.0, 0.50),
    ("gap_frac_over120s", -1.0, 0.40),
    ("gap_cv", -1.0, 0.60),
    ("gap_quantisation", +1.0, 0.50),

    # --- telemetry volume/diversity: browser agents under-generate UI noise --
    ("sum_events_per_action", -1.0, 1.00),
    ("ev_per_action", -1.0, 0.90),
    ("ev_client_per_action", -1.0, 0.90),
    ("ev_client_frac", -1.0, 0.80),
    ("ev_distinct_types_log", -1.0, 0.80),
    ("ev_client_distinct_types_log", -1.0, 0.70),
    ("ev_type_entropy", -1.0, 0.60),
    ("ev_distinct_targets_log", -1.0, 0.70),
    ("ev_target_entropy", -1.0, 0.50),
    ("ev_targets_per_type", -1.0, 0.40),
    ("ev_span_coverage", -1.0, 0.40),

    # --- telemetry cadence: synthetic streams are metronomic ---------------
    ("evgap_cv", -1.0, 0.80),
    ("evgap_burstiness", -1.0, 0.70),
    ("evgap_quantisation", +1.0, 0.70),
    ("evgap_distinct_ratio", -1.0, 0.50),
    ("evgap_round10_share", +1.0, 0.40),

    # --- self-consistency: replayed/synthesised summaries drift ------------
    ("sum_action_mismatch", +1.0, 0.30),
    ("sum_decision_mismatch", +1.0, 0.30),
    ("sum_mean_mismatch", +1.0, 0.30),

    # --- poker behaviour: solver agents snap to canonical sizings ----------
    ("bet_pot_frac_snap_share", +1.0, 0.60),
    ("bet_pot_frac_cv", -1.0, 0.40),
    ("bet_pot_frac_distinct_ratio", -1.0, 0.30),
    ("bet_raise_round100_share", +1.0, 0.30),
    ("sig_pa_entropy", -1.0, 0.40),         # rigid policies repeat patterns
    ("sig_act_repeat_pair_rate", +1.0, 0.30),
    ("sig_act_mean_run_len", +1.0, 0.30),
    ("act_aggression_freq", +1.0, 0.20),    # weak prior, kept small

    # --- vocabulary probes (v1 fallback; on v2 these also fire because the
    #     real names contain the probe tokens: pointer_move, click, ...) -----
    ("ev_pointer_share", -1.0, 0.50),
    ("ev_key_share", -1.0, 0.40),
    ("ev_focus_share", -1.0, 0.40),

    # --- EXACT subject-session.v2 vocabulary (schema-published 2026-07).
    #     Zero variance on v1 batches => zero influence there. -------------
    ("ev2_frac_pointer_move", -1.0, 0.80),   # agents under-produce motion
    ("ev2_move_per_click", -1.0, 0.80),
    ("ev2_moves_per_action", -1.0, 0.70),
    ("ev2_cell_distinct_ratio", -1.0, 0.60),  # humans wander across buckets
    ("ev2_cell_entropy", -1.0, 0.60),
    ("ev2_step_zero_share", +1.0, 0.50),      # sub-bucket/no-op "moves"
    ("ev2_path_over_net", -1.0, 0.50),        # humans overshoot and correct
    ("ev2_dir_reversal_rate", -1.0, 0.40),
    ("ev2_movegap_cv", -1.0, 0.50),           # metronomic mover cadence
    ("ev2_movegap_quantisation", +1.0, 0.50),
    ("ev2_clickgap_cv", -1.0, 0.60),
    ("ev2_clickgap_quantisation", +1.0, 0.60),
    ("ev2_press_cv", -1.0, 0.60),             # button-hold spread is human
    ("ev2_press_quantisation", +1.0, 0.40),
    ("ev2_hover_cv", -1.0, 0.50),             # aim-settle time varies
    ("ev2_frac_scroll", -1.0, 0.30),
    ("ev2_frac_visibility", -1.0, 0.40),      # humans tab away mid-session
    ("ev2_visible_false_share", -1.0, 0.30),
    ("ev2_button_nonzero_share", -1.0, 0.20),  # occasional right-click
)

MIN_BATCH_FOR_RANKS = 4

# Assumed bot prevalence, used only for Brier calibration (the 0.20 term).
# Tournaments are framed as human-vs-browser-agent, so 0.5 is the neutral
# default.  Override with POKER44_V3_PREVALENCE once we observe real windows.
DEFAULT_PREVALENCE = 0.5

# Score band.  Never emit exactly 0 or 1: a confidently wrong extreme is the
# most expensive mistake available under Brier.
SCORE_LO = 0.02
SCORE_HI = 0.98

# Sharpening exponent applied to the centred rank.  >1 pushes scores toward
# the extremes (better Brier when the ranking is good, worse when it is not).
DEFAULT_SHARPEN = 1.25

# Wall-clock budget for one request.  The dev validator's --neuron.timeout
# defaults to 180s but dev DROPPED main's max(30.0, timeout) floor, so a
# validator may legally set a very short timeout.  Once the budget is spent we
# switch remaining sessions to cheap (no telemetry scan) extraction.
DEFAULT_BUDGET_SECONDS = 25.0


def _env_float(name, default):
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _clamp01(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if v != v:
        return 0.5
    return max(0.0, min(1.0, v))


def _otsu_separability(values):
    """Unsupervised bimodality of a sample, in [0, 1].

    KEY IDEA.  Every evaluation window contains BOTH classes -- the validator
    needs both to compute AP at all (``reward`` returns 0 for a single-class
    window unless an env override is set).  So the feature that actually
    separates humans from agents should look BIMODAL across the batch, while a
    feature that carries no class information looks unimodal.  That gives us a
    label-free way to tell which of our priors is live in THIS window.

    Implemented as Otsu's criterion: the fraction of total variance explained
    by the best two-group split, which for a sorted sample is one linear pass.
    Used only to modulate the prior weights, never to replace them -- a
    spuriously bimodal nuisance feature must not be able to take over.
    """
    n = len(values)
    if n < 8:
        return 0.0
    ordered = sorted(values)
    total = sum(ordered)
    mean = total / n
    total_var = sum((v - mean) ** 2 for v in ordered)
    if total_var <= 0:
        return 0.0
    best = 0.0
    left_sum = 0.0
    for i in range(1, n):
        left_sum += ordered[i - 1]
        if ordered[i] == ordered[i - 1]:
            continue
        left_n = i
        right_n = n - i
        left_mean = left_sum / left_n
        right_mean = (total - left_sum) / right_n
        between = left_n * (left_mean - mean) ** 2 + right_n * (right_mean - mean) ** 2
        if between > best:
            best = between
    return max(0.0, min(1.0, best / total_var))


def _average_ranks(values):
    """Average competition ranks mapped to [0, 1]; 0.5 for a constant sample."""
    n = len(values)
    if n < 2:
        return [0.5] * n
    finite = [v for v in values if v == v and abs(v) != float("inf")]
    if not finite or max(finite) == min(finite):
        return [0.5] * n
    order = sorted(range(n), key=lambda i: (values[i], i))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return [r / (n - 1) for r in ranks]


class SessionScorer:
    """Stateless-per-request scorer.  Constructed once, reused per request."""

    def __init__(self, logger=None, artifact_path=None):
        self.log = logger or (lambda level, msg: None)
        self.prevalence = _clamp01(
            _env_float("POKER44_V3_PREVALENCE", DEFAULT_PREVALENCE))
        self.sharpen = max(0.5, _env_float("POKER44_V3_SHARPEN", DEFAULT_SHARPEN))
        self.budget_seconds = max(
            2.0, _env_float("POKER44_V3_BUDGET_SECONDS", DEFAULT_BUDGET_SECONDS))
        self.adaptive = os.getenv("POKER44_V3_ADAPTIVE", "1") != "0"
        # Floor keeps every prior alive even when its feature looks unimodal,
        # so bimodality can amplify but never silence a signal.
        self.adaptive_floor = max(
            0.0, _env_float("POKER44_V3_ADAPTIVE_FLOOR", 0.25))
        self.model = None
        self.model_kind = "rank-ensemble"
        self.artifact_path = artifact_path or os.getenv(
            "POKER44_SESSION_MODEL_ARTIFACT", "")
        self.feature_index = {name: i for i, name in enumerate(sf.FEATURE_NAMES)}

        # Optional second feature bank (staging/telemetry_features.py, 559
        # numpy-backed columns: spectral, permutation/sample entropy, LZ
        # complexity, regression blocks).  It is NOT used by the day-one rank
        # ensemble -- that deliberately stays stdlib-only, fixed-cost and
        # budget-capped -- but it roughly quadruples what a TRAINED model can
        # see, so the trained path consumes both banks concatenated.  Keeping
        # the two banks explicit (rather than merging the files) means the
        # serving fallback can never be broken by a change to the research
        # bank.  Verified: zero feature-name collisions between the two.
        self.extra_bank = None
        self.extra_names = []
        if os.getenv("POKER44_V3_EXTRA_BANK", "1") != "0":
            try:
                import telemetry_features as _tf

                self.extra_bank = _tf.extract_session_features
                self.extra_names = list(_tf.FEATURE_NAMES)
            except Exception:  # noqa: BLE001 - purely optional
                self.extra_bank = None
        self.model_feature_names = list(sf.FEATURE_NAMES) + self.extra_names
        self._load_artifact()

    def _model_row(self, session, base_row):
        """Feature row for the TRAINED path: base bank + optional extra bank."""
        if self.extra_bank is None:
            return base_row
        try:
            _names, extra = self.extra_bank(session)
        except Exception:  # noqa: BLE001
            extra = [0.0] * len(self.extra_names)
        if len(extra) != len(self.extra_names):
            extra = (list(extra) + [0.0] * len(self.extra_names))[
                :len(self.extra_names)]
        return list(base_row) + list(extra)

    # -- optional trained path (empty until we have labelled tournament data)
    def _load_artifact(self):
        if not self.artifact_path or not os.path.exists(self.artifact_path):
            return
        try:
            import joblib

            bundle = joblib.load(self.artifact_path)
            model = bundle.get("model") if isinstance(bundle, dict) else bundle
            names = bundle.get("feature_names") if isinstance(bundle, dict) else None
            if not hasattr(model, "predict_proba"):
                raise TypeError("artifact has no predict_proba")
            if names is not None and list(names) != self.model_feature_names:
                raise ValueError(
                    "artifact feature_names do not match the live banks "
                    "(%d vs %d); refusing to serve a misaligned vector"
                    % (len(names), len(self.model_feature_names)))
            self.model = model
            self.model_kind = "trained-session-model"
            self.log("info", "v3 scorer: loaded session artifact %s"
                     % self.artifact_path)
        except Exception as exc:  # noqa: BLE001 - fall back, never fail startup
            self.model = None
            self.model_kind = "rank-ensemble"
            self.log("warning",
                     "v3 scorer: session artifact %s unusable (%r); "
                     "using the rank ensemble" % (self.artifact_path, exc))

    # -- feature matrix ---------------------------------------------------
    def extract_batch(self, sessions):
        deadline = time.monotonic() + self.budget_seconds
        rows = []
        cheap_from = None
        for i, session in enumerate(sessions):
            cheap = time.monotonic() > deadline
            if cheap and cheap_from is None:
                cheap_from = i
            _names, values = sf.extract(session, cheap=cheap)
            rows.append(values)
        if cheap_from is not None:
            self.log("warning",
                     "v3 scorer: time budget %.1fs exhausted at session %d/%d; "
                     "remaining sessions scored without telemetry features"
                     % (self.budget_seconds, cheap_from, len(sessions)))
        return rows

    # -- scoring paths ----------------------------------------------------
    def _rank_ensemble(self, rows):
        n = len(rows)
        columns = {}
        for name, sign, weight in SIGNALS:
            idx = self.feature_index.get(name)
            if idx is None:
                continue
            values = [row[idx] if idx < len(row) else 0.0 for row in rows]
            if max(values) == min(values):
                continue  # dead signal in this batch -> contributes nothing
            ranks = _average_ranks(values)
            if sign < 0:
                ranks = [1.0 - r for r in ranks]
            effective = weight
            if self.adaptive:
                # Let the window itself say which priors are live.  A flat
                # weighted mean over ~50 signals dilutes a single strongly
                # separating family into insignificance; bimodality weighting
                # fixes that without needing any labels.
                effective = weight * (self.adaptive_floor
                                      + _otsu_separability(values))
            columns[name] = (ranks, effective)
        if not columns:
            return None, 0
        total_w = sum(w for _r, w in columns.values())
        if total_w <= 0:
            return None, 0
        combined = [0.0] * n
        for ranks, weight in columns.values():
            for i in range(n):
                combined[i] += weight * ranks[i]
        combined = [c / total_w for c in combined]
        return combined, len(columns)

    def _absolute_heuristic(self, row):
        """Batch-free backstop for 1..3 session requests.

        Absolute thresholds are far less trustworthy than in-batch ranks --
        we do not know the real scale of the data yet -- so this path stays
        deliberately timid and lands near 0.5.
        """
        def g(name, default=0.0):
            idx = self.feature_index.get(name)
            if idx is None or idx >= len(row):
                return default
            return row[idx]

        votes = []
        cv = g("sum_decision_cv") or g("dt_cv")
        if cv > 0:
            votes.append(1.0 - _clamp01(cv / 0.8))          # low CV -> bot
        epa = g("sum_events_per_action") or g("ev_per_action")
        if epa > 0:
            votes.append(1.0 - _clamp01(epa / 20.0))        # few UI events -> bot
        q = max(g("dt_quantisation"), g("evgap_quantisation"))
        if q > 0:
            votes.append(_clamp01(q * 2.0))
        dr = g("dt_distinct_ratio")
        if dr > 0:
            votes.append(1.0 - _clamp01(dr))
        fast = g("dt_frac_sub300")
        if fast > 0:
            votes.append(_clamp01(fast))
        if not votes:
            return 0.5
        raw = sum(votes) / len(votes)
        return 0.5 + 0.35 * (raw - 0.5) * 2.0 * 0.5  # shrink toward 0.5

    def _calibrate(self, combined):
        """Map combined in-batch scores to calibrated probabilities.

        Rank again (so the output is uniform on ranks), centre on the assumed
        prevalence so roughly the top ``prevalence`` fraction lands above 0.5,
        then apply a mild sharpening.  Ranking (0.80 of the reward) is
        untouched by any of this; only Brier (0.20) is affected.
        """
        ranks = _average_ranks(combined)
        p = self.prevalence
        lo_span = max(1e-6, 1.0 - p)
        hi_span = max(1e-6, p)
        out = []
        for r in ranks:
            if r <= 1.0 - p:
                t = r / lo_span                     # 0..1 within the human band
                value = 0.5 * (t ** self.sharpen)
            else:
                t = (r - (1.0 - p)) / hi_span       # 0..1 within the bot band
                value = 0.5 + 0.5 * (t ** (1.0 / self.sharpen))
            out.append(SCORE_LO + (SCORE_HI - SCORE_LO) * _clamp01(value))
        return out

    # -- public API -------------------------------------------------------
    def score(self, sessions):
        """Return (scores, info).  len(scores) == len(sessions), all in [0,1]."""
        info = {"path": self.model_kind, "n": len(sessions), "active_signals": 0}
        if not sessions:
            return [], info
        try:
            rows = self.extract_batch(sessions)
        except Exception as exc:  # noqa: BLE001
            self.log("error", "v3 scorer: feature extraction failed (%r); "
                              "serving neutral scores" % (exc,))
            info["path"] = "neutral-fallback"
            info["error"] = repr(exc)
            return [0.5] * len(sessions), info

        if self.model is not None:
            try:
                model_rows = [self._model_row(sessions[i], rows[i])
                              for i in range(len(rows))]
                probs = self.model.predict_proba(model_rows)
                scores = [_clamp01(p[1]) for p in probs]
                if len(scores) == len(sessions):
                    info["path"] = "trained-session-model"
                    return scores, info
                raise ValueError("model returned %d scores for %d sessions"
                                 % (len(scores), len(sessions)))
            except Exception as exc:  # noqa: BLE001
                self.log("error", "v3 scorer: trained model failed (%r); "
                                  "falling back to the rank ensemble" % (exc,))
                info["model_error"] = repr(exc)

        if len(sessions) >= MIN_BATCH_FOR_RANKS:
            combined, active = self._rank_ensemble(rows)
            if combined is not None:
                info["path"] = "rank-ensemble"
                info["active_signals"] = active
                return self._calibrate(combined), info
            self.log("warning", "v3 scorer: every sub-signal was constant "
                                "across this batch; using absolute heuristic")

        info["path"] = "absolute-heuristic"
        return [_clamp01(self._absolute_heuristic(row)) for row in rows], info
