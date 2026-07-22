"""Within-request batch transforms that TRAVEL WITH THE ARTIFACT.

sklearn's FunctionTransformer pickles a plain function BY REFERENCE
(module + qualname).  The work2 search used ``work2/transforms2.py``, which is
staging-only, so any artifact containing a non-``rank`` branch could not be
loaded on the serving box.  This module is the deployable home for exactly the
functions the final artifact needs; it must be copied next to
``neurons/live_features.py`` so that ``import poker44_live_tf`` succeeds inside
the miner process.

The function bodies are byte-for-byte the same maths as
``work2/transforms2.py``; ``final/check_tf_equiv.py`` asserts numerical
identity on random and degenerate batches.

Contract: input is the WHOLE incoming batch ``[n_chunks, n_features]`` and the
transform must be defined for any ``n_chunks >= 1`` (production sends exactly
100; training blocks are 70/100/130; a degenerate 1-row batch is possible).
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

EPS = 1e-12


def tf_rank(X):
    """Deployed behaviour: average ranks 1..n within the batch, per column."""
    return rankdata(X, axis=0, method="average")


def tf_rankfrac(X):
    X = np.asarray(X, float)
    return rankdata(X, axis=0, method="average") / X.shape[0]


def tf_qnorm(X):
    X = np.asarray(X, float)
    n = X.shape[0]
    r = rankdata(X, axis=0, method="average")
    return ndtri((r - 0.5) / n)


TRANSFORMS = {
    "rank": tf_rank,
    "rankfrac": tf_rankfrac,
    "qnorm": tf_qnorm,
}
