"""STAGED (not deployed): Poker44 dual-protocol miner for the v3.0 cutover.

WHAT THIS IS
------------
One axon, one port, TWO routes:

    /DetectionSynapse         -> the 468-feature chunk ensemble we serve today
    /SessionDetectionSynapse  -> the v3.0 subject-session scorer

Whichever protocol a validator speaks, we answer.  The flip can happen at any
minute of any day, on some validators before others, and be rolled back --
none of that costs us a round.

WHY IT IS NEEDED (measured, not assumed)
----------------------------------------
Bittensor routes by the synapse CLASS NAME.  Our live axon (UID 226, port
8091) registers only ``/DetectionSynapse``.  Probing it right now with the
v3.0 route returns:

    HTTP 404  {"message":"Synapse name 'SessionDetectionSynapse' not found.
               Available synapses ['Synapse', 'DetectionSynapse']"}

so a v3.0 validator never reaches ``forward()``: ``risk_scores`` comes back
``None``, the validator raises ``ValueError('missing risk_scores')`` and
records reward 0.0.  We do not crash and we do not return wrong data -- we
silently score zero.  With ``moving_average_alpha=0.05`` and a 300s validator
cadence the EMA decays ~0.54x per hour, and weights only ever go to the top 10
positive EMA scores, so we fall out of the paid set within hours.

DEPLOYMENT (do this deliberately; do NOT `git pull` origin/dev onto the live
tree -- ``poker44/validator/synapse.py`` is deleted there, which ImportErrors
``neurons/model_miner.py`` at line 31, and our two local patches sit on files
dev rewrote):

    1. copy   /root/bittensor/poker44-staging/v3/poker44_v3/  ->  repo root
    2. copy   this file -> neurons/dual_miner.py
    3. pm2 restart poker44_miner with the script path pointed at it
       (same CLI args; PYTHONPATH must include the repo root)
    4. verify both routes:
         curl -s -o /dev/null -w '%{http_code}\n' -X POST \
              http://127.0.0.1:8091/DetectionSynapse        -d '{"chunks":[]}'
         curl -s -X POST http://127.0.0.1:8091/SessionDetectionSynapse \
              -d '{"protocol_version":"1","window_id":"x","sessions":[]}'
       403 on the first (blacklist, route exists) and NOT 404 on the second.

HARD CONSTRAINTS BAKED INTO THIS FILE (all verified against bittensor 10.4.1)
  * no ``from __future__ import annotations`` -- ``inspect.signature`` would
    hand ``axon.attach`` the *string* annotation and ``issubclass`` explodes.
  * every attached callback's first parameter must literally be named
    ``synapse`` and be annotated with the exact synapse class.
  * blacklist must return ``typing.Tuple[bool, str]``.  A PEP-585
    ``tuple[bool, str]`` FAILS attach's signature-equality assert (verified:
    ``Tuple[bool,str] == tuple[bool,str]`` is False).
  * ``forward_session`` NEVER raises.  Any exception is a guaranteed 0.0.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Tuple

import bittensor as bt

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), "/root/bittensor/Poker44-subnet",
           "/root/bittensor/poker44-staging/v3"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Survive Finney runtime changes that break metagraph(); must run before the
# neuron is constructed (same as the live miner).
try:
    from poker44.utils import chain_patch

    chain_patch.apply()
except Exception as _exc:  # noqa: BLE001
    bt.logging.warning("chain_patch unavailable: %r" % (_exc,))

from poker44.base.miner import BaseMinerNeuron  # noqa: E402

from poker44_v3.chunk_legacy import LegacyChunkScorer  # noqa: E402
from poker44_v3.protocol_dual import (  # noqa: E402
    DetectionSynapse,
    SessionDetectionSynapse,
)
from poker44_v3.session_capture import SessionCapture  # noqa: E402
from poker44_v3.session_features import N_FEATURES, _strip_forbidden  # noqa: E402
from poker44_v3.session_scorer import SessionScorer  # noqa: E402

MODEL_VERSION = os.getenv("POKER44_V3_MODEL_VERSION", "uid226-dual-1.0")


def _log(level, message):
    getattr(bt.logging, level, bt.logging.info)(message)


class DualProtocolMiner(BaseMinerNeuron):
    """Serves the legacy chunk protocol and the v3.0 session protocol at once."""

    def __init__(self, config=None):
        # BaseMinerNeuron.__init__ already attaches self.forward /
        # self.blacklist / self.priority, which are annotated with
        # DetectionSynapse -> registers the LEGACY route.
        super(DualProtocolMiner, self).__init__(config=config)

        # ---- legacy chunk path (decoupled from neurons/model_miner.py so it
        #      survives a future repo upgrade to the dev tree) --------------
        self.legacy = LegacyChunkScorer(logger=_log)

        # ---- v3.0 session path -------------------------------------------
        self.session_scorer = SessionScorer(logger=_log)
        self.capture = SessionCapture(logger=_log)
        self.session_requests = 0
        self.chunk_requests = 0
        self.first_session_request_at = None

        # ---- ATTACH #2: the v3.0 route ------------------------------------
        self.axon.attach(
            forward_fn=self.forward_session,
            blacklist_fn=self.blacklist_session,
            priority_fn=self.priority_session,
        )
        # The allowlist verify hook is installed by the base class under the
        # class name it knows about only.  Mirror it onto the second route so
        # a configured allowlist protects BOTH, not just one.
        if self.validator_hotkey_whitelist:
            self.axon.verify_fns[SessionDetectionSynapse.__name__] = (
                self.verify_validator_request)
            self.axon.verify_fns[DetectionSynapse.__name__] = (
                self.verify_validator_request)

        # ---- v2 manifest (production still reads it; v3.0 dropped it) ------
        # Without this the legacy route loses the 'transparent' compliance
        # status, so build it exactly as neurons/model_miner.py does.
        self.model_manifest = {}
        try:
            from poker44.utils.model_manifest import (
                build_local_model_manifest,
                evaluate_manifest_compliance,
            )
            from neurons import chunk_features_v3 as _feat_mod

            repo_root = Path("/root/bittensor/Poker44-subnet")
            self.model_manifest = build_local_model_manifest(
                repo_root=repo_root,
                implementation_files=[
                    Path(__file__).resolve(),
                    Path(_feat_mod.__file__).resolve(),
                ],
                defaults={
                    "model_name": "poker44-gbm",
                    "model_version": "3",
                    "framework": "scikit-learn-ensemble",
                    "license": "MIT",
                    "repo_url": "https://github.com/oiid/poker44-gbm",
                    "open_source": True,
                    "inference_mode": "remote",
                    "notes": (
                        "Dual-protocol miner: legacy chunk ensemble over "
                        "chunk_features_v3 plus a v3.0 session handler."
                    ),
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
            status = evaluate_manifest_compliance(self.model_manifest)
            _log("info", "dual miner: manifest status=%s missing=%s"
                 % (status.get("status"), status.get("missing_fields")))
        except Exception as exc:  # noqa: BLE001 - manifest must never break serving
            _log("warning", "dual miner: manifest build failed (%r)" % (exc,))

        _log("info", "dual miner: routes = %s"
             % sorted(self.axon.forward_class_types.keys()))
        _log("info", "dual miner: model_version=%s session_features=%d "
                     "session_scorer=%s legacy_model=%s"
             % (MODEL_VERSION, N_FEATURES, self.session_scorer.model_kind,
                "loaded" if self.legacy.model is not None else "MISSING"))

    # ==================================================================
    # LEGACY ROUTE  ->  /DetectionSynapse
    # ==================================================================
    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []
        self.chunk_requests += 1
        try:
            scores = self.legacy.score(chunks)
            if len(scores) != len(chunks):
                scores = (list(scores) + [0.5] * len(chunks))[:len(chunks)]
        except Exception as exc:  # noqa: BLE001 - never fail a legacy request
            _log("error", "dual miner: legacy scoring failed (%r); neutral" % (exc,))
            scores = [0.5] * len(chunks)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        if self.model_manifest:
            synapse.model_manifest = dict(self.model_manifest)
        if scores:
            _log("info", "Risk scores: n=%d min=%.4f mean=%.4f max=%.4f "
                         "positives=%d/%d"
                 % (len(scores), min(scores), sum(scores) / len(scores),
                    max(scores), sum(s >= 0.5 for s in scores), len(scores)))
        _log("info", "dual miner: legacy route scored %d chunks (req #%d)"
             % (len(chunks), self.chunk_requests))
        return synapse

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)

    # ==================================================================
    # v3.0 ROUTE  ->  /SessionDetectionSynapse
    # ==================================================================
    async def forward_session(
        self, synapse: SessionDetectionSynapse
    ) -> SessionDetectionSynapse:
        """One calibrated bot-risk score per session.  MUST NOT RAISE.

        The reference implementation raises on an unsupported
        ``protocol_version``, on a blank ``window_id`` and on any label-ish
        key found anywhere in the payload.  All three are copied here as
        WARNINGS only.  Rationale: the validator turns any exception into
        reward 0.0, and refusing a request can never score better than
        answering it.  The label check in particular is a live footgun --
        telemetry ``value`` is unconstrained JSON, so a click event carrying a
        button caption (``value={"label": "Raise to 200"}``) is a perfectly
        legal payload that the reference miner would reject outright.
        """
        started = time.monotonic()
        self.session_requests += 1
        if self.first_session_request_at is None:
            self.first_session_request_at = time.time()
            _log("warning",
                 "=== POKER44 v3.0 CUTOVER DETECTED === first "
                 "SessionDetectionSynapse request received "
                 "(window=%s, protocol_version=%s, sessions=%d). Dual-protocol "
                 "miner is answering; legacy route stays up."
                 % (synapse.window_id, synapse.protocol_version,
                    len(synapse.sessions or [])))

        sessions = synapse.sessions or []
        n = len(sessions)
        try:
            if str(synapse.protocol_version) != "1":
                _log("warning",
                     "dual miner: unexpected protocol_version=%r (answering "
                     "anyway; the reference miner would have raised)"
                     % (synapse.protocol_version,))
            if not str(synapse.window_id or "").strip():
                _log("warning", "dual miner: blank window_id (answering anyway)")

            dropped = []
            clean = [_strip_forbidden(s, dropped, "sessions[%d]" % i)
                     for i, s in enumerate(sessions)]
            if dropped:
                _log("warning",
                     "dual miner: stripped %d label-ish key(s) from the payload "
                     "before scoring (never read): %s"
                     % (len(dropped), dropped[:5]))

            scores, info = self.session_scorer.score(clean)
            if len(scores) != n:
                _log("error", "dual miner: scorer returned %d for %d sessions; "
                              "padding to neutral" % (len(scores), n))
                scores = (scores + [0.5] * n)[:n]
            scores = [float(min(1.0, max(0.0, s))) if s == s else 0.5 for s in scores]
        except Exception as exc:  # noqa: BLE001 - a 0.5 vector beats an exception
            _log("error", "dual miner: session scoring blew up (%r); serving "
                          "neutral scores\n%s" % (exc, traceback.format_exc()))
            scores = [0.5] * n
            info = {"path": "exception-fallback", "error": repr(exc)}

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]   # validator ignores this
        synapse.model_version = "%s/%s" % (MODEL_VERSION, info.get("path", "?"))

        try:
            self.capture.record(synapse.window_id, sessions, scores, info)
        except Exception:  # noqa: BLE001
            pass

        elapsed = time.monotonic() - started
        if scores:
            _log("info",
                 "dual miner: v3 route scored %d sessions window=%s path=%s "
                 "signals=%s min=%.4f mean=%.4f max=%.4f in %.2fs (req #%d)"
                 % (n, synapse.window_id, info.get("path"),
                    info.get("active_signals"), min(scores),
                    sum(scores) / len(scores), max(scores), elapsed,
                    self.session_requests))
        return synapse

    async def blacklist_session(
        self, synapse: SessionDetectionSynapse
    ) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority_session(self, synapse: SessionDetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with DualProtocolMiner() as miner:
        bt.logging.info("Dual-protocol Poker44 miner running (legacy + v3.0)...")
        while True:
            bt.logging.info(
                "Miner UID: %s | Incentive: %s | legacy_reqs=%d v3_reqs=%d"
                % (miner.uid, miner.metagraph.I[miner.uid],
                   miner.chunk_requests, miner.session_requests))
            time.sleep(5 * 60)
