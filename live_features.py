"""SHIM -- installed by the 327-column feature-bank deploy.

poker44_v3/chunk_legacy.py hard-codes ``from neurons.live_features import
FEATURE_NAMES, extract_features`` when POKER44_FEATURES=live, so the only way to
serve a wider bank without editing poker44_v3/ is to make that import resolve
here.  The FROZEN 260-column bank this file used to contain now lives, byte for
byte, at neurons/frozen260/live_features.py and is still what columns 0..259
come from.

ROLLBACK: restore neurons/live_features.py from neurons/frozen260/live_features.py
(they are the same file) and put the 260-column artifact back.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from live_features_final import (  # noqa: E402,F401
    FEATURE_NAMES,
    GROUPS,
    N_FEATURES,
    extract_features,
    extract_matrix,
    within_batch_rank,
)
