"""FINAL candidate feature bank: the frozen deployed 260 + 67 admissible new columns.

Column layout is a STRICT PREFIX of the deployed bank:

    columns   0 .. 259   bit-identical to neurons/live_features.py (frozen)
    columns 260 .. 326   the 67 columns of live_features_v2.py that survived
                         (a) the live-shift gate  KS < 0.60 and var_ratio >= 0.05
                         (b) the novelty gate     max |r| vs any frozen column < 0.98

The index list is BAKED IN below (not read from the manifest) so that serving
does not depend on live_features_v2_manifest.json.  ``NOVEL_EXT_INDICES`` are
positions inside live_features_v2's 243-wide extension block, i.e. absolute v2
column = 260 + NOVEL_EXT_INDICES[k].

Files that must travel together for this module to serve:
    live_features_final.py        (this file)
    live_features_v2.py           (the 503-col superset)
    ngram_vocab_v2.json           (Family-A style fixed vocabulary, v2)
    neurons/live_features.py      (frozen 260 bank -- already deployed)
    neurons/ngram_vocab.json      (frozen)
    neurons/live_drop_list.json   (frozen)

live_features_v2 loads the frozen bank from POKER44_LIVE_FEATURES_DIR
(default /root/bittensor/Poker44-subnet/neurons), so the prefix property holds
by construction against whatever is actually deployed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_v2():
    env = os.environ.get("POKER44_V2_FEATURES")
    if env:
        return env
    for cand in (os.path.join(_HERE, "live_features_v2.py"),
                 os.path.join(os.path.dirname(_HERE), "live_features_v2.py")):
        if os.path.exists(cand):
            return cand
    raise RuntimeError("live_features_v2.py not found next to live_features_final.py")


_V2_PATH = _find_v2()

# live_features_v2 loads the FROZEN 260 bank from POKER44_LIVE_FEATURES_DIR.
# When this module is deployed, neurons/live_features.py is replaced by a shim
# that re-exports THIS module -- so v2 must not read that path or it recurses.
# A ``frozen260/`` directory sitting next to this file (a byte copy of the
# frozen bank + its two JSONs) takes priority and breaks the cycle.  In staging
# that directory does not exist and v2 keeps reading the live neurons/ copy,
# which is what makes "prefix == what production emits" true by construction.
_FROZEN_DIR = os.path.join(_HERE, "frozen260")
if os.path.isdir(_FROZEN_DIR) and "POKER44_LIVE_FEATURES_DIR" not in os.environ:
    os.environ["POKER44_LIVE_FEATURES_DIR"] = _FROZEN_DIR


def _load_v2():
    spec = importlib.util.spec_from_file_location("live_features_v2_final", _V2_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("live_features_v2_final", mod)
    spec.loader.exec_module(mod)
    return mod


V2 = _load_v2()
LF = V2.LF
N_BASE = V2.N_BASE
within_batch_rank = V2.within_batch_rank

# --------------------------------------------------------------------------
# 67 admissible-and-novel extension columns (absolute v2 indices).
# Produced by prune_v2_groups.py: keep == True and max_abs_corr_base260 < 0.98.
# --------------------------------------------------------------------------
_NOVEL_ABS = json.load(open(os.path.join(_HERE, "admissible_idx.json")))["novel"] \
    if os.path.exists(os.path.join(_HERE, "admissible_idx.json")) else None

NOVEL_ABS_INDICES = _NOVEL_ABS
if NOVEL_ABS_INDICES is None:
    raise RuntimeError("admissible_idx.json missing next to live_features_final.py")

NOVEL_EXT_INDICES = [i - N_BASE for i in NOVEL_ABS_INDICES]
assert all(0 <= i < V2.N_EXT for i in NOVEL_EXT_INDICES)

FEATURE_NAMES = list(LF.FEATURE_NAMES) + [V2.FEATURE_NAMES[i] for i in NOVEL_ABS_INDICES]
N_FEATURES = len(FEATURE_NAMES)
N_NEW = len(NOVEL_ABS_INDICES)

GROUPS = {k: list(v) for k, v in LF.GROUPS.items()}
GROUPS["base260"] = list(range(N_BASE))
GROUPS["new67"] = list(range(N_BASE, N_FEATURES))

_TAKE = np.asarray(NOVEL_ABS_INDICES, dtype=np.int64)


def extract_features(chunk):
    full = V2.extract_features(chunk)
    return list(full[:N_BASE]) + [full[i] for i in NOVEL_ABS_INDICES]


def extract_matrix(chunks):
    if not chunks:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.asarray([extract_features(c) for c in chunks], dtype=np.float64)


def test_prefix(chunks):
    """Assert columns [0:260) equal the DEPLOYED bank bit-for-bit."""
    n = 0
    for c in chunks:
        a = np.asarray(LF.extract_features(c), dtype=np.float64)
        b = np.asarray(extract_features(c)[:N_BASE], dtype=np.float64)
        if not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
            raise AssertionError("column drift at chunk %d" % n)
        n += 1
    return n
