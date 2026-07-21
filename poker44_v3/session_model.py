"""``BotDetectionModel``-shaped wrapper, for the *other* deployment shape.

Two ways v3.0 can be served:

  A. our dual-protocol neuron (``dual_miner.py``) -- preferred, because it
     keeps the legacy route alive through the cutover;
  B. the subnet's own ``neurons/miner.py`` on origin/dev, which loads a model
     factory from ``POKER44_MODEL_FACTORY="module:create_model"``.

This module makes (B) a one-env-var switch, so if we ever have to run the
upstream neuron we do not have to rewrite the scorer:

    POKER44_MODEL_FACTORY=poker44_v3.session_model:create_model

BOOBY TRAP THIS AVOIDS: README.md on origin/dev tells miners to implement
``predict_bot_risk(self, sessions) -> list[float]``.  That is WRONG.  The real
loader runtime-checks against the ``BotDetectionModel`` Protocol, which
requires ``version``, ``load()`` and ``predict()`` -- a README-shaped model is
rejected with ``TypeError: Poker44 model must expose version, load() and
predict(sessions)``.  Implement ``predict``; ignore the README.

Note that route (B) also puts requests through the reference
``MinerInferenceService``, whose recursive forbidden-key scan RAISES if the
key ``label`` appears anywhere -- including inside the unconstrained telemetry
``value`` blob.  Route (A) does not have that problem.  Prefer route (A).
"""

import os

from poker44_v3.session_scorer import SessionScorer

VERSION = os.getenv("POKER44_V3_MODEL_VERSION", "uid226-dual-1.0")


class Poker44SessionModel:
    """Adapter exposing SessionScorer through the upstream model Protocol."""

    def __init__(self, config=None):
        self.config = config
        self.version = getattr(config, "version", None) or VERSION
        self._scorer = None

    def load(self):
        self._scorer = SessionScorer()
        return None

    def predict(self, sessions):
        if self._scorer is None:
            self.load()
        scores, _info = self._scorer.score(list(sessions or []))
        return scores


def create_model(config=None):
    return Poker44SessionModel(config)
