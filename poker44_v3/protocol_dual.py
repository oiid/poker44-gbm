"""Both Poker44 wire protocols in one module, so one axon can serve both.

WHY THIS FILE EXISTS
--------------------
Bittensor derives the HTTP route from the *Python class name*:

    axon.attach(...)  ->  self.router.add_api_route(path=f"/{param_class.__name__}")
    dendrite POSTs    ->  http://{ip}:{port}/{synapse.__class__.__name__}

(verified in bittensor 10.4.1: core/axon.py attach(), core/dendrite.py).

v3.0 renamed the synapse ``DetectionSynapse`` -> ``SessionDetectionSynapse``,
so a v3.0 validator POSTs to ``/SessionDetectionSynapse``.  The dev repo's
``DetectionSynapse = SessionDetectionSynapse`` alias is a *Python* alias only;
``DetectionSynapse.__name__`` is still ``'SessionDetectionSynapse'`` so the
alias buys ZERO wire compatibility.

Therefore we define two genuinely distinct classes with the two distinct
names and attach BOTH to the same axon.  Verified live: one axon, one port,
routes ``['DetectionSynapse', 'SessionDetectionSynapse', 'Synapse']``, both
answering HTTP 200.

DO NOT add ``from __future__ import annotations`` to this module or to any
module whose functions get attached.  ``inspect.signature`` does not evaluate
string annotations, so ``axon.attach`` would see the *string* ``'MySyn'``
instead of the class and blow up on ``issubclass``.  (Verified.)

FIELD-ORDER WARNING: ``Synapse.body_hash`` hashes ``required_hash_fields`` in
class-declared order and the axon re-computes it in ``verify_body_integrity``.
The tuple below must stay byte-identical to the validator's
(``poker44/protocol.py`` on origin/dev: protocol_version, window_id, sessions).
"""

from typing import Any, ClassVar, Dict, List, Optional

import bittensor as bt
from pydantic import ConfigDict, Field


class DetectionSynapse(bt.Synapse):
    """LEGACY (production today, spec_version 1).

    Byte-identical to ``poker44/validator/synapse.py`` on main.  Serves the
    route ``/DetectionSynapse``.  Keep this class untouched: any change to
    field names, order or ``required_hash_fields`` breaks body-hash
    verification against validators still on main.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: List[List[dict]] = Field(default_factory=list)
    risk_scores: Optional[List[float]] = None
    predictions: Optional[List[bool]] = None
    model_manifest: Optional[Dict[str, Any]] = None

    required_hash_fields: ClassVar[List[str]] = ["chunks"]

    def deserialize(self) -> "DetectionSynapse":
        return self


class SessionDetectionSynapse(bt.Synapse):
    """v3.0 (origin/dev, spec_version 2).  Serves ``/SessionDetectionSynapse``.

    Mirrors ``poker44/protocol.py`` on origin/dev exactly.  ``model_manifest``
    is deliberately absent -- it was deleted from the protocol.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protocol_version: str = "1"
    window_id: str = ""
    sessions: List[Dict[str, Any]] = Field(default_factory=list)

    risk_scores: Optional[List[float]] = None
    predictions: Optional[List[bool]] = None
    model_version: Optional[str] = None

    required_hash_fields: ClassVar[List[str]] = [
        "protocol_version",
        "window_id",
        "sessions",
    ]

    def deserialize(self) -> "SessionDetectionSynapse":
        return self


# Sanity guards: these two invariants are the whole point of the file.
assert DetectionSynapse.__name__ == "DetectionSynapse"
assert SessionDetectionSynapse.__name__ == "SessionDetectionSynapse"
