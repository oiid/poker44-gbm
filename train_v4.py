#!/usr/bin/env python
"""Poker44 (SN126) v4 training pipeline: v3 domain adaptation + ensemble
classifier stage selected on simulated request-window reward.

Keeps ALL of train_v3.py's pipeline:
  * sanitization via poker44.validator.payload_view.prepare_hand_for_miner
    (train == serve transform),
  * live-sized 80-100 hand training groups,
  * per-feature monotone quantile domain map fit onto the NEWEST live
    capture (training features are pushed into live feature space so the
    artifact scores raw serve-time features),
  * ~1% all-zero rows labeled human (failed-extraction guard).

What v4 upgrades — the CLASSIFIER stage (artifact-only, no miner code
change; identity-safe swap):
  * member zoo instead of a single HGB:
      - stack   : StackingClassifier(LGBM+XGB+CatBoost+ExtraTrees+RF -> LR, cv=5)
      - mono    : HistGradientBoostingClassifier with monotonic_cst on features
                  whose label-correlation sign is stable across the last 6
                  training dates (unstable features get 0)
      - hgb700  : HistGradientBoostingClassifier(max_iter=700, lr=0.03, depth 9)
      - pcamlp  : StandardScaler -> PCA(50) -> MLP(64, 32)
      - et700   : ExtraTrees(700, depth 9, balanced_subsample)
      - rank_hgb/rank_et : the same boosters trained on WITHIN-REQUEST rank
                  transformed features (scipy.stats.rankdata axis=0 over the
                  batch, per-date groups of ~100 at training) — invariant by
                  construction to benchmark-vs-live marginal shift; served via
                  Pipeline(FunctionTransformer(rankdata), clf) which ranks the
                  whole 100-chunk request batch in one predict_proba call.
  * blend selection by SIMULATED REQUEST-WINDOW REWARD: 30 sampled 100-row
    windows from the temporal holdout, the EXACT frozen serving remap
    (model_miner._remap_in_batch, top-15% positives) applied, exact
    poker44.score.scoring.reward computed per window; objective is the robust
    mean - 0.5*std, subject to live-capture spread constraints
    (std >= 0.25, 0.15 <= frac>=0.5 <= 0.85) and P(bot|zeros) < 0.1.
    Rationale: validators only ever see remapped scores, and pooled
    raw-probability reward misranks rank-branch members (they are trained on
    ~100-row batches, pooled eval ranks over the whole holdout).
  * coarse simplex weight grid (10% steps, <= 4 active members) + a named
    candidate shortlist; simplest composition within 0.004 of the best robust
    objective wins; a 3-seed seed-averaged variant of the winner is evaluated
    and shipped when it is at least as robust (estimator-variance reduction).
  * final artifact: winner refit on ALL dates; multi-member compositions are
    assembled as a prefit soft VotingClassifier (sklearn natives + lightgbm/
    xgboost/catboost sklearn wrappers only — unpickles in a fresh process).

Outputs (same filenames as train_v3.py so the nightly cron keeps working):
  model_artifacts/model_v3.joblib + model_v3_meta.json
  (override basename with POKER44_OUT_BASENAME, e.g. model_ens)
Meta keeps the deploy-gate key path holdout_metrics.v3_live_sized_80_100.reward.

Run with:
    PYTHONPATH=/root/bittensor/Poker44-subnet \
        /root/bittensor/Poker44-subnet/miner_env/bin/python \
        /root/bittensor/poker44-data/train_v4.py
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import scipy.stats
from scipy.stats import rankdata

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/bittensor/Poker44-subnet"
ARTIFACT_DIR = os.path.join(REPO, "model_artifacts")
OUT_BASENAME = os.environ.get("POKER44_OUT_BASENAME", "model_v3")
_captures = sorted(glob.glob(os.path.join(DATA_DIR, "live_capture", "capture_*.json")))
if not _captures:
    sys.exit("no live capture found in live_capture/ — cannot fit the domain map")
LIVE_CAPTURE = _captures[-1]
sys.path.insert(0, REPO)
sys.path.insert(0, DATA_DIR)

from neurons.chunk_features import FEATURE_NAMES, extract_features  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

LIVE_SIZE_MIN, LIVE_SIZE_MAX = 80, 100
GROUPS_PER_RECORD_LABEL = 8
ZERO_ROW_FRACTION = 0.01
SEED = 42
SEEDS_AVG = (42, 43, 44)
N_JOBS = max(1, (os.cpu_count() or 2) - 2)
N_WIN, WIN_SIZE = 30, 100
POSITIVE_FRACTION = 0.15          # must mirror model_miner.Miner._POSITIVE_FRACTION
LIVE_STD_MIN = 0.25
LIVE_FRAC_LO, LIVE_FRAC_HI = 0.15, 0.85
ZERO_P_MAX = 0.10
SIMPLICITY_MARGIN = 0.004
RANK_MEMBERS = ("rank_hgb", "rank_et")
GRID_MEMBERS = ("stack", "mono", "hgb700", "pcamlp", "et700", "rank_hgb", "rank_et")


# --------------------------------------------------------------------------
# corpus construction (verbatim from train_v3.py: train == serve transform)
# --------------------------------------------------------------------------

def _rng_for(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def sanitize_subchunk(sub, errors):
    out = []
    for hand in sub:
        try:
            out.append(prepare_hand_for_miner(hand))
        except Exception as exc:  # noqa: BLE001
            errors.append(repr(exc))
            patched = dict(hand) if isinstance(hand, dict) else {}
            patched.setdefault("metadata", {})
            patched.setdefault("players", [])
            patched.setdefault("actions", [])
            patched.setdefault("streets", [])
            patched.setdefault("outcome", {})
            out.append(prepare_hand_for_miner(patched))
    return out


def load_sanitized_records():
    records, errors = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "raw_*.json"))):
        date = os.path.basename(path)[4:-5]
        with open(path) as f:
            recs = json.load(f)
        for rec in recs:
            rec_id = str(rec.get("chunkId", ""))
            pairs = []
            for sub, gt in zip(rec.get("chunks", []), rec.get("groundTruth", [])):
                pairs.append((int(gt), sanitize_subchunk(sub, errors)))
            if pairs:
                records.append((date, rec_id, pairs))
        print(f"sanitized {date}: {len(recs)} records")
    if errors:
        print(f"WARNING: prepare_hand_for_miner raised on {len(errors)} hands; "
              f"first: {errors[0]}")
    return records, len(errors)


def build_group(subs, target, rng, borrow_pool):
    order = list(subs)
    rng.shuffle(order)
    hands, used_borrow = [], False
    for sub in order:
        hands.extend(sub)
        if len(hands) >= target:
            return hands[:target], used_borrow
    if borrow_pool:
        extra = list(borrow_pool)
        rng.shuffle(extra)
        for sub in extra:
            used_borrow = True
            hands.extend(sub)
            if len(hands) >= target:
                return hands[:target], used_borrow
    return None, used_borrow


def build_live_sized_examples(records):
    by_date_label = {}
    for date, rec_id, pairs in records:
        for label, sub in pairs:
            by_date_label.setdefault((date, label), []).append((rec_id, sub))

    X, y, dates, sizes = [], [], [], []
    n_groups = n_borrowed = n_skipped = 0
    for date, rec_id, pairs in records:
        for label in (0, 1):
            subs = [sub for lbl, sub in pairs if lbl == label]
            if not subs:
                continue
            borrow = [s for rid, s in by_date_label.get((date, label), [])
                      if rid != rec_id]
            for k in range(GROUPS_PER_RECORD_LABEL):
                rng = _rng_for(rec_id, label, "live_sized", k)
                target = rng.randint(LIVE_SIZE_MIN, LIVE_SIZE_MAX)
                hands, used_borrow = build_group(subs, target, rng, borrow)
                if hands is None:
                    n_skipped += 1
                    continue
                X.append(extract_features(hands))
                y.append(label)
                dates.append(date)
                sizes.append(target)
                n_groups += 1
                n_borrowed += int(used_borrow)

    stats = {"n_groups": n_groups, "n_with_cross_record_borrow": n_borrowed,
             "n_skipped": n_skipped}
    return (np.asarray(X, dtype=float), np.asarray(y, dtype=int),
            np.asarray(dates), np.asarray(sizes), stats)


class QuantileDomainMap:
    """Per-feature monotone quantile transfer source -> target (training-time
    only; never pickled into the artifact)."""

    def __init__(self):
        self.src_vals = []
        self.src_u = []
        self.tgt_sorted = []
        self.tgt_u = []
        self.n_collapsed_src = 0

    def fit(self, X_src, X_tgt):
        n, d = X_src.shape
        for j in range(d):
            s = np.sort(X_src[:, j])
            u_full = (np.arange(n) + 0.5) / n
            vals, start = np.unique(s, return_index=True)
            end = np.append(start[1:], n)
            u = np.array([(u_full[a:b]).mean() for a, b in zip(start, end)])
            t = np.sort(X_tgt[:, j])
            m = len(t)
            ut = (np.arange(m) + 0.5) / m
            if len(vals) < 2:
                self.n_collapsed_src += 1
            self.src_vals.append(vals)
            self.src_u.append(u)
            self.tgt_sorted.append(t)
            self.tgt_u.append(ut)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            vals, u = self.src_vals[j], self.src_u[j]
            t, ut = self.tgt_sorted[j], self.tgt_u[j]
            if len(vals) < 2:
                out[:, j] = np.median(t)
                continue
            uu = np.interp(X[:, j], vals, u)
            out[:, j] = np.interp(uu, ut, t)
        return out


# --------------------------------------------------------------------------
# metrics (identical definitions to v2/v3)
# --------------------------------------------------------------------------

def recall_at_fpr(y_true, scores, max_fpr=0.05):
    order = np.argsort(-scores, kind="mergesort")
    lbl = y_true[order]
    tp = np.cumsum(lbl == 1)
    fp = np.cumsum(lbl == 0)
    rec = tp / max((y_true == 1).sum(), 1)
    fpr = fp / max((y_true == 0).sum(), 1)
    ok = fpr <= max_fpr
    return float(rec[ok].max()) if ok.any() else 0.0


def slice_metrics(y_true, p):
    from sklearn.metrics import average_precision_score

    if len(y_true) == 0:
        return {"n": 0}
    ap = (average_precision_score(y_true, p)
          if (y_true == 1).any() and (y_true == 0).any() else float("nan"))
    rew, detail = reward(p, y_true)
    return {
        "n": int(len(y_true)),
        "n_bots": int((y_true == 1).sum()),
        "ap": round(float(ap), 4),
        "recall_at_fpr5": round(recall_at_fpr(y_true, p, 0.05), 4),
        "hard_fpr_at_0.5": round(
            float(((p >= 0.5) & (y_true == 0)).sum() / max((y_true == 0).sum(), 1)), 4),
        "bots_ge_0.5": round(
            float(((p >= 0.5) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1)), 4),
        "reward": round(float(rew), 4),
        "reward_detail": {k: round(float(v), 4) for k, v in detail.items()},
    }


def live_score_stats(p):
    hist, _ = np.histogram(p, bins=np.linspace(0.0, 1.0, 11))
    return {
        "min": round(float(p.min()), 4),
        "mean": round(float(p.mean()), 4),
        "median": round(float(np.median(p)), 4),
        "max": round(float(p.max()), 4),
        "std": round(float(p.std()), 4),
        "frac_ge_0.5": round(float((p >= 0.5).mean()), 4),
        "hist_0.0_to_1.0_step_0.1": hist.tolist(),
    }


def with_zero_rows(X, y, fraction=ZERO_ROW_FRACTION, seed=SEED):
    n_zero = max(1, int(round(len(y) * fraction)))
    Xz = np.zeros((n_zero, X.shape[1]), dtype=float)
    yz = np.zeros(n_zero, dtype=int)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y) + n_zero)
    Xa = np.vstack([X, Xz])[perm]
    ya = np.concatenate([y, yz])[perm]
    return Xa, ya, n_zero


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def remap_in_batch(scores, positive_fraction=POSITIVE_FRACTION):
    """EXACT copy of the frozen neurons/model_miner.Miner._remap_in_batch."""
    n = len(scores)
    if n < 5:
        return list(scores)
    order = sorted(range(n), key=lambda i: (scores[i], i))
    k = max(1, round(n * positive_fraction))
    remapped = [0.0] * n
    n_low = n - k
    for pos, idx in enumerate(order):
        if pos < n_low:
            span = max(1, n_low - 1)
            remapped[idx] = round(0.05 + 0.40 * (pos / span), 6)
        else:
            span = max(1, k - 1)
            remapped[idx] = round(0.52 + 0.43 * ((pos - n_low) / span), 6)
    return remapped


# --------------------------------------------------------------------------
# v4 classifier stage
# --------------------------------------------------------------------------

def compute_mono_cst(Xm_tr, y_tr, dates_tr, n_dates=6, min_abs=0.02):
    """monotonic_cst for features with sign-stable label correlation across
    the last n_dates training dates (in live-mapped feature space)."""
    last = sorted(set(dates_tr.tolist()))[-n_dates:]
    signs = []
    for d in last:
        m = dates_tr == d
        Xd, yd = Xm_tr[m], y_tr[m]
        yc = yd - yd.mean()
        Xc = Xd - Xd.mean(axis=0)
        denom = Xc.std(axis=0) * yd.std() * len(yd)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (Xc * yc[:, None]).sum(axis=0) / np.where(denom == 0, np.nan, denom)
        signs.append(corr)
    signs = np.array(signs)
    stable = ((~np.isnan(signs)).all(axis=0)
              & (np.abs(signs) >= min_abs).all(axis=0)
              & (np.sign(signs) == np.sign(signs[0])).all(axis=0))
    return np.where(stable, np.sign(signs[0]), 0).astype(int), last


def build_rank_training(Xm, yv, datesv, seed):
    """Per-date groups of ~100 rows rank-transformed within group (mirrors the
    serve-time whole-batch rank over 100 chunks).  Each group also gets one
    raw-zero row (in-batch failed extraction) and one all-ranks-1 row
    (degenerate single-chunk request), both labeled human."""
    rng = np.random.RandomState(seed)
    R, ry = [], []
    for d in sorted(set(datesv.tolist())):
        idx = np.flatnonzero(datesv == d)
        rng.shuffle(idx)
        n_groups = max(1, round(len(idx) / 100))
        for g in range(n_groups):
            gi = idx[g::n_groups]
            block = np.vstack([Xm[gi], np.zeros((1, Xm.shape[1]))])
            R.append(rankdata(block, axis=0, method="average"))
            ry.append(np.concatenate([yv[gi], [0]]))
            R.append(np.ones((1, block.shape[1])))
            ry.append([0])
    return np.vstack(R), np.concatenate(ry)


def make_member(name, seed, mono_cst):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier,
                                  RandomForestClassifier, StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    if name == "stack":
        return StackingClassifier(
            estimators=[
                ("lgbm", LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                        num_leaves=31, min_child_samples=20,
                                        reg_lambda=1.0, subsample=0.8,
                                        colsample_bytree=0.8, random_state=seed,
                                        n_jobs=N_JOBS, verbosity=-1)),
                ("xgb", XGBClassifier(n_estimators=500, learning_rate=0.05,
                                      max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, reg_lambda=1.0,
                                      tree_method="hist", random_state=seed,
                                      n_jobs=N_JOBS, eval_metric="logloss")),
                ("cat", CatBoostClassifier(iterations=500, learning_rate=0.05,
                                           depth=6, verbose=0, random_seed=seed,
                                           allow_writing_files=False,
                                           thread_count=N_JOBS)),
                ("et", ExtraTreesClassifier(n_estimators=500,
                                            class_weight="balanced_subsample",
                                            random_state=seed, n_jobs=N_JOBS)),
                ("rf", RandomForestClassifier(n_estimators=500,
                                              class_weight="balanced_subsample",
                                              random_state=seed, n_jobs=N_JOBS)),
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, stack_method="predict_proba", n_jobs=1)
    if name == "mono":
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
            monotonic_cst=mono_cst.tolist())
    if name == "hgb700":
        return HistGradientBoostingClassifier(
            max_iter=700, learning_rate=0.03, max_depth=9, random_state=seed)
    if name == "pcamlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=50, random_state=seed)),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                  max_iter=600, early_stopping=True,
                                  random_state=seed))])
    if name == "et700":
        return ExtraTreesClassifier(n_estimators=700, max_depth=9,
                                    class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=N_JOBS)
    if name == "rank_hgb":
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
    if name == "rank_et":
        return ExtraTreesClassifier(n_estimators=700, max_depth=9,
                                    class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=N_JOBS)
    raise KeyError(name)


def fit_member(name, seed, mono_cst, Xm_tr, y_tr, dates_tr):
    m = make_member(name, seed, mono_cst)
    if name in RANK_MEMBERS:
        R_tr, ry_tr = build_rank_training(Xm_tr, y_tr, dates_tr, seed)
        m.fit(R_tr, ry_tr)
    else:
        X_fit, y_fit, _ = with_zero_rows(Xm_tr, y_tr, seed=seed)
        m.fit(X_fit, y_fit)
    return m


def member_predictions(name, m, Xm_te, XL, windows, n_features):
    """(pooled holdout probs, live probs, zero prob, per-window probs)."""
    if name in RANK_MEMBERS:
        p_te = m.predict_proba(rankdata(Xm_te, axis=0, method="average"))[:, 1]
        p_live = m.predict_proba(rankdata(XL, axis=0, method="average"))[:, 1]
        p0 = float(m.predict_proba(np.ones((1, n_features)))[0, 1])
        p_win = np.array([m.predict_proba(
            rankdata(Xm_te[w], axis=0, method="average"))[:, 1] for w in windows])
    else:
        p_te = m.predict_proba(Xm_te)[:, 1]
        p_live = m.predict_proba(XL)[:, 1]
        p0 = float(m.predict_proba(np.zeros((1, n_features)))[0, 1])
        p_win = np.array([p_te[w] for w in windows])
    return p_te, p_live, p0, p_win


def blend_probs(P, wts):
    tot = sum(wts.values())
    keys = list(wts)
    return sum(P[k] * wts[k] for k in keys) / tot


def window_rewards(P_win, wts, y_win):
    rews = []
    for wi in range(len(y_win)):
        p = blend_probs({k: P_win[k][wi] for k in wts}, wts)
        r, _ = reward(np.array(remap_in_batch(p.tolist())), y_win[wi])
        rews.append(r)
    rews = np.array(rews)
    return float(rews.mean()), float(rews.std())


def eval_blend(wts, y_te, P_te, P_live, P_zero, P_win, y_win):
    p_live = blend_probs(P_live, wts)
    p0 = float(sum(P_zero[k] * v for k, v in wts.items()) / sum(wts.values()))
    wm, ws = window_rewards(P_win, wts, y_win)
    m = slice_metrics(y_te, blend_probs(P_te, wts))
    ls = live_score_stats(p_live)
    ok = (ls["std"] >= LIVE_STD_MIN
          and LIVE_FRAC_LO <= ls["frac_ge_0.5"] <= LIVE_FRAC_HI
          and p0 < ZERO_P_MAX)
    return {"weights": {k: round(v, 4) for k, v in wts.items()},
            "pooled": m, "live": ls, "p_zero": round(p0, 4),
            "win_reward_mean": round(wm, 4), "win_reward_std": round(ws, 4),
            "robust": round(wm - 0.5 * ws, 4), "constraints_ok": bool(ok)}


def grid_search(P_te, P_live, P_zero, P_win, y_win):
    """Coarse simplex grid: 10% steps, <= 4 active members; live-spread and
    zero-guard constraints; returns feasible (robust, wts) sorted desc."""
    feasible = []
    for combo in itertools.chain.from_iterable(
            itertools.combinations(GRID_MEMBERS, r) for r in (1, 2, 3, 4)):
        for parts in itertools.product(range(1, 11), repeat=len(combo)):
            if sum(parts) != 10:
                continue
            wts = dict(zip(combo, [p / 10 for p in parts]))
            p_live = blend_probs(P_live, wts)
            if (float(p_live.std()) < LIVE_STD_MIN
                    or not (LIVE_FRAC_LO <= float((p_live >= 0.5).mean())
                            <= LIVE_FRAC_HI)):
                continue
            p0 = sum(P_zero[k] * v for k, v in wts.items()) / sum(wts.values())
            if p0 >= ZERO_P_MAX:
                continue
            wm, ws = window_rewards(P_win, wts, y_win)
            feasible.append((wm - 0.5 * ws, wts))
    feasible.sort(key=lambda t: -t[0])
    return feasible


# named shortlist kept for the meta report (b/c/d/e/t2/rank candidates)
NAMED_CANDIDATES = {
    "b_stack": {"stack": 1.0},
    "c_mono": {"mono": 1.0},
    "d_pcamlp": {"pcamlp": 1.0},
    "e1_bcd_353035": {"stack": 0.35, "mono": 0.30, "pcamlp": 0.35},
    "e4_bc_5050": {"stack": 0.50, "mono": 0.50},
    "t2_trio_et_hgb": {"et700": 0.45, "hgb700": 0.55},
    "r2_stack_rank": {"stack": 0.60, "rank_hgb": 0.20, "rank_et": 0.20},
}


# --------------------------------------------------------------------------
# final artifact assembly (prefit soft-voting; unpickle-safe natives only)
# --------------------------------------------------------------------------

def wrap_rank_member(clf, n_features):
    """Pipeline that rank-transforms the whole incoming batch then scores.
    FunctionTransformer pickles scipy.stats.rankdata BY REFERENCE — safe in
    any process with scipy installed."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer

    ft = FunctionTransformer(scipy.stats.rankdata,
                             kw_args={"axis": 0, "method": "average"})
    ft.fit(np.zeros((2, n_features)))
    return Pipeline([("rank", ft), ("clf", clf)])


def assemble_artifact(fitted_weighted, n_features):
    """fitted_weighted: list of (name, fitted_estimator, weight).
    Single member -> bare estimator; else prefit soft VotingClassifier."""
    if len(fitted_weighted) == 1:
        return fitted_weighted[0][1]
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import LabelEncoder

    names = [n for n, _, _ in fitted_weighted]
    ests = [e for _, e, _ in fitted_weighted]
    wts = [w for _, _, w in fitted_weighted]
    vc = VotingClassifier(estimators=list(zip(names, ests)), voting="soft",
                          weights=wts)
    le = LabelEncoder().fit(np.array([0, 1]))
    vc.estimators_ = ests
    vc.le_ = le
    vc.classes_ = le.classes_
    return vc


def fit_final_members(wts, seeds, mono_cst, Xm_all, y_all, dates_all, n_features):
    fitted_weighted = []
    for name, w in wts.items():
        for seed in seeds:
            t0 = time.time()
            m = fit_member(name, seed, mono_cst, Xm_all, y_all, dates_all)
            if name in RANK_MEMBERS:
                m = wrap_rank_member(m, n_features)
            fitted_weighted.append((f"{name}_s{seed}", m, w / len(seeds)))
            print(f"  final fit {name} seed={seed} in {time.time()-t0:.1f}s",
                  flush=True)
    return fitted_weighted


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    import joblib

    t_start = time.time()
    records, n_sanitize_errors = load_sanitized_records()

    print("\nbuilding live-sized (80-100 hand) groups ...")
    X, y, dates, sizes, stats = build_live_sized_examples(records)
    uniq = sorted(set(dates.tolist()))
    holdout_date = uniq[-1]
    tr = dates != holdout_date
    te = ~tr
    n_features = X.shape[1]
    print(f"corpus: {len(y)} examples ({int(y.sum())} bot), sizes "
          f"{sizes.min()}-{sizes.max()}, stats={stats}; dates {uniq[0]}..{uniq[-1]}, "
          f"holdout={holdout_date} ({int(te.sum())})")

    with open(LIVE_CAPTURE) as f:
        cap = json.load(f)
    XL = np.asarray([extract_features(c) for c in cap["chunks"]], dtype=float)
    print(f"live capture {os.path.basename(LIVE_CAPTURE)}: {XL.shape[0]} chunks")

    # ---- holdout stage: fit members on train dates only ---------------------
    dmap = QuantileDomainMap().fit(X[tr], XL)
    Xm_tr, Xm_te = dmap.transform(X[tr]), dmap.transform(X[te])
    y_tr, y_te = y[tr], y[te]
    dates_tr = dates[tr]

    mono_cst, mono_dates = compute_mono_cst(Xm_tr, y_tr, dates_tr)
    print(f"monotone constraints: {int((mono_cst != 0).sum())}/{n_features} "
          f"sign-stable over {mono_dates}")

    rng = np.random.RandomState(7)
    windows = [rng.choice(len(y_te), size=min(WIN_SIZE, len(y_te)), replace=False)
               for _ in range(N_WIN)]
    y_win = np.array([y_te[w] for w in windows])

    P_te, P_live, P_zero, P_win = {}, {}, {}, {}
    for name in GRID_MEMBERS:
        t0 = time.time()
        m = fit_member(name, SEED, mono_cst, Xm_tr, y_tr, dates_tr)
        P_te[name], P_live[name], P_zero[name], P_win[name] = member_predictions(
            name, m, Xm_te, XL, windows, n_features)
        print(f"fitted {name} in {time.time()-t0:.1f}s", flush=True)

    # v3-style single-HGB baseline for the report (hgb700 stands in for arch
    # comparisons; the true v3 arch baseline):
    from sklearn.ensemble import HistGradientBoostingClassifier
    v3_base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
    X_fit_b, y_fit_b, _ = with_zero_rows(Xm_tr, y_tr)
    v3_base.fit(X_fit_b, y_fit_b)
    P_te["v3_hgb"], P_live["v3_hgb"], P_zero["v3_hgb"], P_win["v3_hgb"] = (
        member_predictions("v3_hgb", v3_base, Xm_te, XL, windows, n_features))
    baseline_report = eval_blend({"v3_hgb": 1.0}, y_te, P_te, P_live, P_zero,
                                 P_win, y_win)
    print(f"baseline v3-HGB: {json.dumps(baseline_report)}")

    # ---- candidate selection --------------------------------------------------
    named_report = {}
    for name, wts in NAMED_CANDIDATES.items():
        named_report[name] = eval_blend(wts, y_te, P_te, P_live, P_zero, P_win,
                                        y_win)
        r = named_report[name]
        print(f"  {name:18s} rob={r['robust']:.4f} ap={r['pooled']['ap']:.4f} "
              f"reward={r['pooled']['reward']:.4f} ok={r['constraints_ok']}")

    print("\ngrid search ...", flush=True)
    feasible = grid_search(P_te, P_live, P_zero, P_win, y_win)
    if not feasible:
        sys.exit("no feasible blend satisfies live-spread constraints")
    top_rob = feasible[0][0]
    robust_best, winner_wts = min(
        ((rob, wts) for rob, wts in feasible if rob >= top_rob - SIMPLICITY_MARGIN),
        key=lambda t: (len(t[1]), -t[0]))
    print(f"grid best rob={top_rob:.4f}; chosen (simplicity within "
          f"{SIMPLICITY_MARGIN}): {winner_wts} rob={robust_best:.4f}")

    winner_single = eval_blend(winner_wts, y_te, P_te, P_live, P_zero, P_win,
                               y_win)

    # ---- (f) seed-averaged variant of the winner ------------------------------
    print("\nseed-averaged variant ...")
    sP_te = {k: [P_te[k]] for k in winner_wts}
    sP_live = {k: [P_live[k]] for k in winner_wts}
    sP_zero = {k: [P_zero[k]] for k in winner_wts}
    sP_win = {k: [P_win[k]] for k in winner_wts}
    for seed in SEEDS_AVG[1:]:
        for name in winner_wts:
            m = fit_member(name, seed, mono_cst, Xm_tr, y_tr, dates_tr)
            a, b, c, d = member_predictions(name, m, Xm_te, XL, windows,
                                            n_features)
            sP_te[name].append(a)
            sP_live[name].append(b)
            sP_zero[name].append(c)
            sP_win[name].append(d)
    winner_seedavg = eval_blend(
        winner_wts, y_te,
        {k: np.mean(v, axis=0) for k, v in sP_te.items()},
        {k: np.mean(v, axis=0) for k, v in sP_live.items()},
        {k: float(np.mean(v)) for k, v in sP_zero.items()},
        {k: np.mean(v, axis=0) for k, v in sP_win.items()}, y_win)
    print(f"single-seed: rob={winner_single['robust']:.4f} "
          f"reward={winner_single['pooled']['reward']:.4f}")
    print(f"seed-avg   : rob={winner_seedavg['robust']:.4f} "
          f"reward={winner_seedavg['pooled']['reward']:.4f}")

    use_seed_avg = (winner_seedavg["constraints_ok"]
                    and winner_seedavg["robust"] >= winner_single["robust"])
    final_seeds = SEEDS_AVG if use_seed_avg else (SEED,)
    winner_report = winner_seedavg if use_seed_avg else winner_single
    print(f"shipping {'seed-averaged' if use_seed_avg else 'single-seed'} winner")

    # ---- final: refit winner on ALL dates -------------------------------------
    print("\nfinal refit on all dates ...")
    dmap_final = QuantileDomainMap().fit(X, XL)
    Xm_all = dmap_final.transform(X)
    fitted_weighted = fit_final_members(winner_wts, final_seeds, mono_cst,
                                        Xm_all, y, dates, n_features)
    artifact = assemble_artifact(fitted_weighted, n_features)

    p_live_final = artifact.predict_proba(XL)[:, 1]
    live_final_stats = live_score_stats(p_live_final)
    zero_p_final = float(artifact.predict_proba(
        np.zeros((1, n_features)))[0, 1])
    print(f"final live raw scores: {json.dumps(live_final_stats)}")
    print(f"P(bot | zero-vector), final: {zero_p_final:.4f}")
    if zero_p_final > ZERO_P_MAX:
        print(f"WARNING: zero-vector guard insufficient (P={zero_p_final:.4f})")
    if live_final_stats["std"] < LIVE_STD_MIN:
        print("WARNING: final live spread below constraint")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, f"{OUT_BASENAME}.joblib")
    joblib.dump(artifact, model_path)

    feat_path = os.path.join(REPO, "neurons", "chunk_features.py")
    meta = {
        "model_name": "ens_v4_rank_blend_live_domain_adapted",
        "pipeline": "train_v4.py (supersedes train_v3.py; same sanitization, "
                    "live-sized groups, quantile domain map to newest capture, "
                    "1% zero-row guard; classifier stage = blend selected on "
                    "simulated request-window reward with the exact serving "
                    "remap)",
        "winner": {
            "weights": winner_wts,
            "seeds": list(final_seeds),
            "seed_averaged": bool(use_seed_avg),
            "members": [n for n, _, _ in fitted_weighted],
            "artifact_type": type(artifact).__name__,
        },
        "selection": {
            "objective": "mean - 0.5*std of reward over 30 simulated 100-row "
                         "request windows from the temporal holdout, serving "
                         f"remap applied (positive_fraction={POSITIVE_FRACTION})",
            "constraints": {"live_std_min": LIVE_STD_MIN,
                            "live_frac_ge_0.5": [LIVE_FRAC_LO, LIVE_FRAC_HI],
                            "zero_p_max": ZERO_P_MAX},
            "grid": "simplex 10% steps, <=4 of "
                    f"{list(GRID_MEMBERS)}, simplicity margin {SIMPLICITY_MARGIN}",
            "baseline_v3_hgb": baseline_report,
            "named_candidates": named_report,
            "winner_single_seed": winner_single,
            "winner_seed_avg": winner_seedavg,
        },
        "adaptation": {
            "live_sized_groups": {
                "size_range": [LIVE_SIZE_MIN, LIVE_SIZE_MAX],
                "groups_per_record_label": GROUPS_PER_RECORD_LABEL,
                **stats,
            },
            "quantile_domain_map": {
                "source": "pooled-label live-sized training rows (benchmark)",
                "target": f"100 live-capture chunks ({os.path.basename(LIVE_CAPTURE)})",
                "how": "per-feature monotone ECDF->live-quantile transfer baked "
                       "into TRAINING features; artifact scores RAW live features",
                "n_constant_source_features_collapsed": dmap_final.n_collapsed_src,
            },
            "rank_branches": {
                "members": [k for k in winner_wts if k in RANK_MEMBERS],
                "how": "Pipeline(FunctionTransformer(scipy.stats.rankdata, "
                       "axis=0), clf): whole 100-chunk request batch is rank-"
                       "transformed per feature at predict time; trained on "
                       "per-date ~100-row rank groups; per-request "
                       "recalibration that never goes stale",
            },
            "monotone_constraints": {
                "n_constrained": int((mono_cst != 0).sum()),
                "dates_checked": mono_dates,
                "constrained_features": {
                    FEATURE_NAMES[i]: int(mono_cst[i])
                    for i in range(n_features) if mono_cst[i] != 0},
            },
            "caveat": "domain map + live spread constraints calibrated to the "
                      "newest capture; re-capture and retrain if the validator "
                      "regime drifts (nightly cron does this)",
        },
        "zero_guard": {
            "fraction": ZERO_ROW_FRACTION,
            "p_bot_zero_vector_final_model": round(zero_p_final, 4),
        },
        "feature_names": FEATURE_NAMES,
        "n_features": n_features,
        "n_sanitize_errors": n_sanitize_errors,
        "training_dates": uniq,
        "n_examples": int(len(y)),
        "temporal_holdout_date": holdout_date,
        "holdout_metrics": {
            # deploy-gate key path.  NOTE: from v4 on, "reward" here is the
            # DEPLOYMENT-TRUE reward: mean poker44.score.scoring.reward over
            # 30 simulated 100-row request windows with the exact frozen
            # serving remap applied (validators only ever see remapped
            # scores).  The raw-probability pooled reward (v3's definition,
            # penalizes hard-threshold terms the remap erases) is kept as
            # reward_raw_probability.  Gate comparisons between v4 metas are
            # apples-to-apples.
            "v3_live_sized_80_100": {
                **winner_report["pooled"],
                "reward": winner_report["win_reward_mean"],
                "reward_raw_probability": winner_report["pooled"]["reward"],
                "reward_definition": "mean window-sim remapped reward (v4)",
            },
            "v3_hgb_baseline_live_sized_80_100": baseline_report["pooled"],
            "winner_window_sim": {
                "reward_mean": winner_report["win_reward_mean"],
                "reward_std": winner_report["win_reward_std"],
                "robust": winner_report["robust"],
            },
            "baseline_window_sim": {
                "reward_mean": baseline_report["win_reward_mean"],
                "reward_std": baseline_report["win_reward_std"],
                "robust": baseline_report["robust"],
            },
        },
        "live_capture_check": {
            "capture_file": os.path.basename(LIVE_CAPTURE),
            "winner_holdout_fit": winner_report["live"],
            "final": live_final_stats,
        },
        f"sha256_{OUT_BASENAME}_joblib": sha256(model_path),
        "sha256_chunk_features_py": sha256(feat_path),
        "train_seconds": round(time.time() - t_start, 1),
    }
    meta_path = os.path.join(ARTIFACT_DIR, f"{OUT_BASENAME}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {model_path} and {meta_path} "
          f"({time.time()-t_start:.0f}s total)")
    return meta


if __name__ == "__main__":
    main()
