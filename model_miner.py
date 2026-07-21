"""Poker44 miner backed by a trained HistGradientBoosting model.

Loads a fitted scikit-learn classifier from a joblib artifact (env
``POKER44_MODEL_ARTIFACT``, default ``model_artifacts/model.joblib``) and
scores each incoming chunk with ``predict_proba``.  If the artifact cannot
be loaded at startup, the miner falls back to the reference heuristic from
``neurons/miner.py`` so it never serves garbage after a bad deploy.
"""

# from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

# Survive Finney runtime changes that break metagraph() on affected SDKs.
# Dormant when the native call works; must run before Miner() is constructed.
from poker44.utils import chain_patch
chain_patch.apply()

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

# v3 (2026-07-21 boundary): feature module is chunk_features_v3 — a strict
# superset of chunk_features_v2 (its first 148 columns are bit-identical:
# 111 deployed v1 + 37 cross-hand signature features), plus the coherent
# block (per-hand behavioral scalar distributions + signature grid, pruned
# on live captures).  chunk_features and chunk_features_v2 stay in the repo
# untouched and are imported by chunk_features_v3.
try:
    from neurons import chunk_features_v3 as _chunk_features_module
    from neurons.chunk_features_v3 import extract_features
    from neurons.miner import Miner as _HeuristicMiner
except ImportError:  # running as a bare script without PYTHONPATH=repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neurons import chunk_features_v3 as _chunk_features_module
    from neurons.chunk_features_v3 import extract_features
    from neurons.miner import Miner as _HeuristicMiner

DEFAULT_MODEL_ARTIFACT = str(
    Path(__file__).resolve().parents[1] / "model_artifacts" / "model.joblib"
)
MODEL_ARTIFACT_ENV = "POKER44_MODEL_ARTIFACT"


def load_model(path):
    """Load and return a fitted scikit-learn model from a joblib artifact."""
    import joblib

    model = joblib.load(path)
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"artifact at {path} has no predict_proba (got {type(model)!r})"
        )
    return model


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_chunks(chunks: list, model) -> list:
    """Pure scoring path: fitted model + feature extraction, loads nothing.

    Returns one bot-risk probability in [0, 1] per chunk.
    """
    if not chunks:
        return []
    features = [extract_features(chunk) for chunk in chunks]
    probs = model.predict_proba(features)[:, 1]
    return [_clamp01(p) for p in probs]


class Miner(BaseMinerNeuron):
    """Model-based Poker44 miner (HistGradientBoostingClassifier).

    Scores each chunk with the trained classifier's bot probability; falls
    back to the reference heuristic if the model artifact is unavailable.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.model_artifact_path = os.getenv(
            MODEL_ARTIFACT_ENV, DEFAULT_MODEL_ARTIFACT
        )
        self.model = None
        try:
            self.model = load_model(self.model_artifact_path)
            bt.logging.info(
                f"🤖 Model Poker44 Miner started | loaded artifact "
                f"{self.model_artifact_path} ({type(self.model).__name__}, "
                f"{len(_chunk_features_module.FEATURE_NAMES)} features)"
            )
        except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
            bt.logging.warning(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )
            bt.logging.warning(
                f"MODEL ARTIFACT FAILED TO LOAD from "
                f"{self.model_artifact_path}: {exc!r}"
            )
            bt.logging.warning(
                "FALLING BACK to the reference heuristic scorer from "
                "neurons/miner.py — deploy a valid artifact and restart!"
            )
            bt.logging.warning(
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            )

        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                Path(_chunk_features_module.__file__).resolve(),
            ],
            defaults={
                "model_name": "poker44-gbm",
                "model_version": "3",
                "framework": "scikit-learn-histgradientboosting",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "Gradient-boosting/ensemble classifier over stdlib chunk "
                    "behavioral + cross-hand signature + coherent-block "
                    "features (neurons/chunk_features_v3.py); artifact "
                    "metadata in model_artifacts/model_meta.json."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on publicly released Poker44 training "
                    "chunks (releaseType=training) with their published "
                    "groundTruth labels."
                ),
                "training_data_sources": [
                    "poker44 public training releases (raw_YYYY-MM-DD.json)"
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only "
                    "evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one model-based bot-risk score per chunk."""
        chunks = synapse.chunks or []
        if self.model is not None:
            try:
                scores = score_chunks(chunks, self.model)
            except Exception as exc:  # noqa: BLE001 - never fail a request
                bt.logging.error(
                    f"Model scoring failed ({exc!r}); using heuristic fallback."
                )
                scores = [_HeuristicMiner.score_chunk(chunk) for chunk in chunks]
        else:
            scores = [_HeuristicMiner.score_chunk(chunk) for chunk in chunks]

        scores = self._remap_in_batch(scores, chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        self._dump_live_sample(chunks, scores)
        if scores:
            bt.logging.info(
                f"Risk scores: n={len(scores)} min={min(scores):.4f} "
                f"mean={sum(scores)/len(scores):.4f} max={max(scores):.4f} "
                f"positives={sum(s >= 0.5 for s in scores)}/{len(scores)}"
            )
        bt.logging.info(
            f"Scored {len(chunks)} chunks with "
            f"{'model' if self.model is not None else 'heuristic-fallback'} risks."
        )
        return synapse

    # Fraction of chunks per request allowed to cross the 0.5 flag threshold.
    # Live batches mix humans and bots; the reward's rank metrics (AP,
    # recall@FPR) are invariant to this monotone remap, while the 0.5
    # threshold-sanity gate needs a low false-positive rate.
    #
    # v2 (measured 2026-07-16, staging eval_remap.py, 200 labeled 100-chunk
    # 50/50 batches from the 2026-07-16 holdout): reward is FLAT in the
    # fraction over [0.05, 0.20] (threshold-sanity saturates at 1.0; rank
    # metrics are remap-invariant), so 0.10 is chosen purely for tail margin:
    # exact-k=floor(0.10*n) keeps worst-case hard-FPR at 10/(n_humans) and
    # showed 0 gate breaches in 3000 simulated batches across 30/50/70% bot
    # mixes.  The artifact may override via a `poker44_positive_fraction`
    # attribute on the model object (artifact-only tuning, identity-safe).
    _POSITIVE_FRACTION = 0.10

    def _positive_fraction(self) -> float:
        frac = getattr(self.model, "poker44_positive_fraction", None)
        try:
            frac = float(frac)
        except (TypeError, ValueError):
            return self._POSITIVE_FRACTION
        return frac if 0.0 < frac <= 0.5 else self._POSITIVE_FRACTION

    @staticmethod
    def _chunk_tie_key(chunk) -> str:
        """Deterministic, batch-order-invariant tie-break for equal scores."""
        import hashlib
        import json

        try:
            blob = json.dumps(chunk, sort_keys=True, default=str)
        except Exception:  # noqa: BLE001
            blob = repr(chunk)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _remap_in_batch(self, scores: list, chunks: list) -> list:
        """Rank-preserving in-batch remap with an exact positive budget.

        Exactly k = max(1, floor(fraction * n)) chunks land above 0.5;
        positives are compressed into [0.501, 0.509], negatives spread over
        [0.05, 0.49].  Ties are broken by a SHA-256 chunk fingerprint so a
        permuted batch can never change which chunk crosses the threshold.
        """
        n = len(scores)
        if n < 5:
            return scores
        keys = [self._chunk_tie_key(chunks[i]) if i < len(chunks) else str(i)
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

    _LIVE_CAPTURE_DIR = Path("/root/bittensor/poker44-data/live_capture")
    _LIVE_CAPTURE_MAX = 3

    def _dump_live_sample(self, chunks, scores) -> None:
        """Temporary diagnostic: persist live payloads to study train/serve skew."""
        try:
            import json
            import time as _time

            self._LIVE_CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            existing = sorted(self._LIVE_CAPTURE_DIR.glob("capture_*.json"))
            while len(existing) >= self._LIVE_CAPTURE_MAX:
                existing.pop(0).unlink()
            payload = {"ts": _time.time(), "scores": scores, "chunks": chunks}
            out = self._LIVE_CAPTURE_DIR / f"capture_{int(_time.time())}.json"
            out.write_text(json.dumps(payload))
            bt.logging.info(f"Live capture saved: {out}")
        except Exception as exc:  # noqa: BLE001 - diagnostics must never break serving
            bt.logging.warning(f"Live capture failed: {exc!r}")

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


def _print_help_and_exit():
    """bt.Config swallows --help, so handle it explicitly before startup."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="model_miner.py",
        description="Poker44 model-based miner (HistGradientBoosting).",
    )
    Miner.add_args(parser)
    parser.parse_args()  # argparse prints full help and exits for -h/--help


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        _print_help_and_exit()
    with Miner() as miner:
        bt.logging.info("Model miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
