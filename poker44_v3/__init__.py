"""Poker44 v3.0 (subject-session) staging package for UID 226.

Nothing in here is deployed.  Copy the package to the repo root and point
pm2 at ``neurons/dual_miner.py`` to cut over.

Modules
    protocol_dual    both wire synapses (DetectionSynapse + SessionDetectionSynapse)
    session_features stdlib feature extraction over v3.0 subject sessions
    session_scorer   day-one in-batch rank ensemble + fallback chain
    session_adapter  v3.0 session -> legacy chunk shape (research/offline)
    session_capture  payload capture + telemetry-vocabulary distillation
    session_model    BotDetectionModel-compatible wrapper + factory
"""

__all__ = [
    "protocol_dual",
    "session_features",
    "session_scorer",
    "session_adapter",
    "session_capture",
    "session_model",
]

__version__ = "1.0.0"
