"""Legacy chunk scoring, decoupled from neurons/model_miner.py.

WHY NOT JUST IMPORT THE LIVE MINER?
-----------------------------------
``neurons/model_miner.py`` does ``from poker44.validator.synapse import
DetectionSynapse`` at module scope -- and origin/dev DELETES that file.  It
also imports ``poker44.utils.model_manifest``; ``model_manifest`` is gone from
the v3.0 protocol entirely.  So the moment the repo is upgraded, importing the
live miner raises ImportError and we would lose the legacy route at exactly
the moment we most need both routes alive.

This module therefore depends only on things that survive the upgrade:
  * ``neurons/chunk_features_v3.py`` -- untracked, ours, stdlib-only, imports
    nothing from the ``poker44`` package;
  * the joblib artifact in ``model_artifacts/``;
  * joblib itself.

The in-batch remap is copied verbatim (behaviour-identical) from
``neurons/model_miner.py`` rather than imported, for the same reason.  Its
tuning rationale lives in that file's comments; the numbers here must not be
changed independently of it.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

DEFAULT_ARTIFACT = "/root/bittensor/Poker44-subnet/model_artifacts/model.joblib"

# Fraction of chunks per request allowed above the 0.5 flag threshold.
# Verbatim from neurons/model_miner.py::_POSITIVE_FRACTION.
POSITIVE_FRACTION = 0.10


def _clamp01(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.5
    if v != v:
        return 0.5
    return max(0.0, min(1.0, v))


class LegacyChunkScorer:
    """The deployed 468-feature ensemble, importable without the live miner."""

    def __init__(self, artifact_path=None, logger=None):
        self.log = logger or (lambda level, msg: None)
        self.artifact_path = (
            artifact_path
            or os.getenv("POKER44_MODEL_ARTIFACT", DEFAULT_ARTIFACT))
        self.model = None
        self.extract_features = None
        self.n_features = 0
        self._load()

    def _load(self):
        for candidate in ("/root/bittensor/Poker44-subnet",
                          str(Path(__file__).resolve().parents[2])):
            if candidate not in sys.path:
                sys.path.append(candidate)
        try:
            # POKER44_FEATURES selects the feature bank so rollback is an env
            # flip, not a re-edit.  "live" = the 260-feature live-robust
            # n-gram bank (0% live-collapsed, selected under the live-proxy
            # harness); "v3" = the legacy 468-feature bank (83% live-collapsed).
            # The bank and the artifact are ONE UNIT — switching one without
            # the other feeds the wrong width and scores garbage.
            if os.getenv("POKER44_FEATURES", "v3").strip().lower() == "live":
                from neurons.live_features import FEATURE_NAMES, extract_features
            else:
                from neurons.chunk_features_v3 import FEATURE_NAMES, extract_features

            self.extract_features = extract_features
            self.n_features = len(FEATURE_NAMES)
        except Exception as exc:  # noqa: BLE001
            self.log("error", "legacy scorer: chunk_features_v3 unavailable "
                              "(%r); the legacy route will serve neutral"
                     % (exc,))
            return
        try:
            import joblib

            model = joblib.load(self.artifact_path)
            if not hasattr(model, "predict_proba"):
                raise TypeError("artifact has no predict_proba: %r" % type(model))
            self.model = model
            self.log("info", "legacy scorer: %s loaded (%d features)"
                     % (self.artifact_path, self.n_features))
        except Exception as exc:  # noqa: BLE001
            self.log("warning", "legacy scorer: artifact %s unusable (%r)"
                     % (self.artifact_path, exc))

    def _positive_fraction(self):
        frac = getattr(self.model, "poker44_positive_fraction", None)
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return POSITIVE_FRACTION
        return frac if 0.0 < frac <= 0.5 else POSITIVE_FRACTION

    @staticmethod
    def _tie_key(chunk):
        try:
            blob = json.dumps(chunk, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            blob = repr(chunk)
        return hashlib.sha256(blob.encode()).hexdigest()

    def remap_in_batch(self, scores, chunks):
        """Rank-preserving in-batch remap with an exact positive budget.

        Verbatim behaviour from neurons/model_miner.py::_remap_in_batch.

        MODE (env POKER44_REMAP_MODE, default "raw" since 2026-07-21):
          * "raw"     -- return the model's own probabilities untouched.  The
            production reward switched to 0.50*AP_skill + 0.30*recall@FPR5 +
            0.20*BrierSkill, which has NO 0.5-threshold term and instead pays
            for genuine calibration.  The exact-k band ([0.05,0.509], batch
            mean ~0.29 against a ~50/50 prevalence) scores Brier WORSE than
            the predict-prevalence baseline, so BrierSkill clips to 0 and the
            whole 20% component is forfeited.  Measured on simulated windows:
            raw beats exact-k by +0.033 reward at moderate ranking quality and
            +0.034 at strong, and is never worse (BrierSkill is floored at 0).
          * "exact_k" -- the legacy band, optimal under the OLD formula whose
            threshold_sanity term required FPR@0.5 <= 0.10.
        Ranking (AP, recall@FPR) is invariant to any monotone map, so this
        switch only ever moves the calibration component.
        """
        if os.getenv("POKER44_REMAP_MODE", "raw").strip().lower() == "raw":
            return list(scores)
        n = len(scores)
        if n < 5:
            return scores
        keys = [self._tie_key(chunks[i]) if i < len(chunks) else str(i)
                for i in range(n)]
        order = sorted(range(n), key=lambda i: (scores[i], keys[i]))
        k = max(1, int(n * self._positive_fraction()))
        remapped = [0.0] * n
        n_low = n - k
        for pos, idx in enumerate(order):
            if pos < n_low:
                span = max(1, n_low - 1)
                remapped[idx] = round(0.05 + 0.44 * (pos / span), 6)
            else:
                span = max(1, k - 1)
                remapped[idx] = round(0.501 + 0.008 * ((pos - n_low) / span), 6)
        return remapped

    def score(self, chunks):
        """One bot-risk score per chunk.  Never raises."""
        if not chunks:
            return []
        if self.model is None or self.extract_features is None:
            return [0.5] * len(chunks)
        try:
            features = [self.extract_features(c) for c in chunks]
            probs = self.model.predict_proba(features)[:, 1]
            scores = [_clamp01(p) for p in probs]
            if any(math.isnan(s) for s in scores):
                raise ValueError("model produced NaN")
            return self.remap_in_batch(scores, chunks)
        except Exception as exc:  # noqa: BLE001
            self.log("error", "legacy scorer failed (%r); serving neutral"
                     % (exc,))
            return [0.5] * len(chunks)
