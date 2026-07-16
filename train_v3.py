#!/usr/bin/env python
"""Poker44 (SN126) v3 training pipeline: domain-adapted bot detection.

Why v3 (diagnosis from live capture capture_1783985513.json, 2026-07-14):
  v2 scored ALL 100 live validator chunks in [0.76, 0.99] because the live
  payload domain differs radically from the sanitized benchmark domain:
    * game regime: live is ~100bb uniform stacks / 2.8-8.2bb pots / ~1bb
      bets; benchmark is ~236bb stacks / 8-126bb pots / 23-126bb bets.
    * action-count window: every live hand shows 5-8 actions; benchmark
      hands include 2-4-action hands and 12x-duplicated single-action
      hands, so chunk-level min/max action counts differ structurally
      (``vis_actions_per_hand_min`` alone inflated v2's live mean by 0.165).
    * amount quantization: live nonzero amounts take only ~60 distinct
      values (~1bb buckets +- hash noise) vs ~1200 in the benchmark, so all
      ``*_n_distinct_per_hand`` / ``*_distinct_ratio`` / ``*_cv`` features
      sit far BELOW the benchmark-bot mean, i.e. live looks "hyper-bot".
    * table mix: live chunks interleave 6/7/8/9-max tables (80-100 hands);
      benchmark is 6-max only (30-40 hand sub-chunks).
  These shifts are in nuisance directions the benchmark cannot reproduce by
  regrouping alone, so v3 combines:
    1. LIVE-SIZED GROUPS: training groups of 80-100 hands (matching the
       live hands-per-chunk distribution), built by concatenating whole
       shuffled same-label sub-chunks of the same record (borrowing
       same-label sub-chunks from same-date records only when short).
    2. QUANTILE DOMAIN MAP (CORAL-style, rank-preserving): every feature of
       every training row is pushed through a monotone per-feature quantile
       transfer fit from the pooled (both-label) training marginal onto the
       100-chunk live-capture marginal.  Because the map is monotone, all
       within-benchmark rank relations (= the label signal that survives
       the "pattern_hardened_v2" marginal equalization) are preserved,
       while irreparably-shifted features (e.g. constant-in-live
       ``vis_actions_per_hand_min``, ``pot_before_bb_max``) collapse to
       constants and are ignored by the trees.  The map is baked into the
       TRAINING features, so the artifact stays a bare
       HistGradientBoostingClassifier and the deployed miner
       (neurons/model_miner.py) needs no code or import changes: live
       features are scored RAW because the model is trained in live
       feature space.
    3. Same HGB hyperparameters as v2 and the same ~1% all-zero rows
       labeled human (guard rows appended AFTER the mapping, since a failed
       serve-time extraction yields literal zeros).

Artifacts: model_artifacts/model_v3.joblib + model_v3_meta.json.
Does NOT touch model.joblib and does not restart anything.

Run with:
    PYTHONPATH=/root/bittensor/Poker44-subnet \
        /root/bittensor/Poker44-subnet/miner_env/bin/python \
        /root/bittensor/poker44-data/train_v3.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import sys

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/bittensor/Poker44-subnet"
ARTIFACT_DIR = os.path.join(REPO, "model_artifacts")
# Always calibrate the domain map to the newest captured live batch.
_captures = sorted(glob.glob(os.path.join(DATA_DIR, "live_capture", "capture_*.json")))
if not _captures:
    sys.exit("no live capture found in live_capture/ — cannot fit the domain map")
LIVE_CAPTURE = _captures[-1]
sys.path.insert(0, REPO)
sys.path.insert(0, DATA_DIR)

from neurons.chunk_features import FEATURE_NAMES, extract_features  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

LIVE_SIZE_MIN, LIVE_SIZE_MAX = 80, 100  # observed live hands-per-chunk range
GROUPS_PER_RECORD_LABEL = 8
ZERO_ROW_FRACTION = 0.01
SEED = 42


# --------------------------------------------------------------------------
# corpus construction (sanitize identical to v2: train == serve transform)
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
    """Concatenate whole shuffled same-label sub-chunks to >= target hands,
    truncate to target (v2 strategy).  Returns (hands, used_borrow) or
    (None, False)."""
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
    """GROUPS_PER_RECORD_LABEL groups of 80-100 hands per (record, label)."""
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


def build_native_examples(records, only_dates=None):
    """Native 30-40 hand sub-chunks (for reference evaluation)."""
    X, y, dates = [], [], []
    for date, rec_id, pairs in records:
        if only_dates is not None and date not in only_dates:
            continue
        for label, sub in pairs:
            X.append(extract_features(sub))
            y.append(label)
            dates.append(date)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), np.asarray(dates)


# --------------------------------------------------------------------------
# quantile domain map: source (benchmark) marginals -> live-capture marginals
# --------------------------------------------------------------------------

class QuantileDomainMap:
    """Per-feature monotone quantile transfer source -> target.

    Only used at TRAINING time (the artifact is a bare classifier trained in
    live feature space), so this class never needs to be importable by the
    miner process.
    """

    def __init__(self):
        self.src_vals = []   # per-feature sorted unique source values
        self.src_u = []      # per-feature ECDF position of each unique value
        self.tgt_sorted = [] # per-feature sorted target values
        self.tgt_u = []      # per-feature ECDF grid of target
        self.n_collapsed_src = 0

    def fit(self, X_src, X_tgt):
        n, d = X_src.shape
        for j in range(d):
            s = np.sort(X_src[:, j])
            u_full = (np.arange(n) + 0.5) / n
            vals, start = np.unique(s, return_index=True)
            # average ECDF position over ties
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
            if len(vals) < 2:  # constant source feature -> live median
                out[:, j] = np.median(t)
                continue
            uu = np.interp(X[:, j], vals, u)
            out[:, j] = np.interp(uu, ut, t)
        return out


# --------------------------------------------------------------------------
# metrics (identical definitions to v2)
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
    hist, edges = np.histogram(p, bins=np.linspace(0.0, 1.0, 11))
    return {
        "min": round(float(p.min()), 4),
        "mean": round(float(p.mean()), 4),
        "median": round(float(np.median(p)), 4),
        "max": round(float(p.max()), 4),
        "std": round(float(p.std()), 4),
        "frac_ge_0.5": round(float((p >= 0.5).mean()), 4),
        "hist_0.0_to_1.0_step_0.1": hist.tolist(),
    }


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def make_model(seed=SEED):
    from sklearn.ensemble import HistGradientBoostingClassifier

    # same family + hyperparameters as v1/v2
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
    )


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


def main():
    import joblib

    records, n_sanitize_errors = load_sanitized_records()

    print("\nbuilding live-sized (80-100 hand) groups ...")
    X, y, dates, sizes, stats = build_live_sized_examples(records)
    uniq = sorted(set(dates.tolist()))
    holdout_date = uniq[-1]
    tr = dates != holdout_date
    te = ~tr
    print(f"corpus: {len(y)} examples ({int(y.sum())} bot / {int((y == 0).sum())} "
          f"human), sizes {sizes.min()}-{sizes.max()}, stats={stats}; "
          f"dates {uniq[0]}..{uniq[-1]}, holdout={holdout_date} ({int(te.sum())})")

    with open(LIVE_CAPTURE) as f:
        cap = json.load(f)
    XL = np.asarray([extract_features(c) for c in cap["chunks"]], dtype=float)
    print(f"live capture: {XL.shape[0]} chunks (recorded v2 scores "
          f"min={min(cap['scores']):.4f} mean={np.mean(cap['scores']):.4f} "
          f"max={max(cap['scores']):.4f})")

    # ---- holdout evaluation ------------------------------------------------
    dmap = QuantileDomainMap().fit(X[tr], XL)
    print(f"domain map: {dmap.n_collapsed_src} constant source features "
          f"collapsed to live median")
    Xm_tr = dmap.transform(X[tr])
    Xm_te = dmap.transform(X[te])

    X_fit, y_fit, n_zero_tr = with_zero_rows(Xm_tr, y[tr])
    model = make_model()
    model.fit(X_fit, y_fit)
    p_te = model.predict_proba(Xm_te)[:, 1]

    # v2 deployed reference on identical holdout rows.  NOTE: model.joblib
    # was refit on ALL dates including the holdout date, so these numbers
    # are IN-SAMPLE-inflated; the fair baseline is v2style_raw_holdout_fit.
    v2 = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
    p_te_v2 = v2.predict_proba(X[te])[:, 1]

    # fair ablation baseline: same rows/hyperparams/zero-guard, RAW features
    raw_model = make_model()
    Xr_fit, yr_fit, _ = with_zero_rows(X[tr], y[tr])
    raw_model.fit(Xr_fit, yr_fit)
    p_te_raw = raw_model.predict_proba(X[te])[:, 1]

    # native 30-40 holdout slice (mapped for v3, raw for v2)
    Xn, yn, _ = build_native_examples(records, only_dates={holdout_date})
    p_n = model.predict_proba(dmap.transform(Xn))[:, 1]
    p_n_v2 = v2.predict_proba(Xn)[:, 1]

    holdout_metrics = {
        "v3_live_sized_80_100": slice_metrics(y[te], p_te),
        "v2style_raw_holdout_fit_live_sized_80_100": slice_metrics(y[te], p_te_raw),
        "v2_deployed_IN_SAMPLE_live_sized_80_100": slice_metrics(y[te], p_te_v2),
        "v3_native_30_40": slice_metrics(yn, p_n),
        "v2_deployed_IN_SAMPLE_native_30_40": slice_metrics(yn, p_n_v2),
    }
    print(f"\nholdout {holdout_date} (v3 trained on {len(y_fit)} rows incl "
          f"{n_zero_tr} zero-guard rows):")
    for name, m in holdout_metrics.items():
        print(f"  {name}: {json.dumps(m)}")

    # ---- live-capture raw-score check --------------------------------------
    p_live_v3_holdout = model.predict_proba(XL)[:, 1]
    p_live_v2 = v2.predict_proba(XL)[:, 1]
    p_live_raw = raw_model.predict_proba(XL)[:, 1]
    print(f"\nlive raw scores, v3(holdout-fit): "
          f"{json.dumps(live_score_stats(p_live_v3_holdout))}")
    print(f"live raw scores, v2style raw:     "
          f"{json.dumps(live_score_stats(p_live_raw))}")
    print(f"live raw scores, v2 (deployed):   "
          f"{json.dumps(live_score_stats(p_live_v2))}")

    zero_p_holdout = float(model.predict_proba(np.zeros((1, X.shape[1])))[0, 1])
    print(f"P(bot | zero-vector), holdout-trained v3: {zero_p_holdout:.4f}")

    # ---- final artifact: refit map + model on ALL dates ---------------------
    dmap_final = QuantileDomainMap().fit(X, XL)
    Xm_all = dmap_final.transform(X)
    X_all, y_all, n_zero_all = with_zero_rows(Xm_all, y)
    final = make_model()
    final.fit(X_all, y_all)
    zero_p_final = float(final.predict_proba(np.zeros((1, X.shape[1])))[0, 1])
    p_live_final = final.predict_proba(XL)[:, 1]
    live_final_stats = live_score_stats(p_live_final)
    print(f"\nP(bot | zero-vector), final v3: {zero_p_final:.4f}")
    print(f"live raw scores, v3 FINAL: {json.dumps(live_final_stats)}")
    if zero_p_final > 0.4:
        print("WARNING: zero-vector guard insufficient (P > 0.4)")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACT_DIR, "model_v3.joblib")
    joblib.dump(final, model_path)

    feat_path = os.path.join(REPO, "neurons", "chunk_features.py")
    meta = {
        "model_name": "hgb_v3_live_domain_adapted",
        "diagnosis": {
            "v2_live_failure": "all 100 live chunks scored 0.76-0.99 raw",
            "root_cause": "live payload regime (100bb stacks, 2.8-8.2bb pots, "
                          "~1bb coarse amounts with ~60 distinct values, 5-8 "
                          "action window on every hand, 6-9max table interleave, "
                          "80-100 hand chunks) lies outside the sanitized "
                          "benchmark support (236bb stacks, 8-126bb pots, ~1200 "
                          "distinct amounts, 2-12 visible actions, 6max only); "
                          "distinctness/cv features shift in the bot-ish "
                          "direction so HGB extrapolates every live chunk to "
                          "high P(bot). Largest single driver: "
                          "vis_actions_per_hand_min (mean drop 0.165 when "
                          "substituted with benchmark-human mean).",
        },
        "adaptation": {
            "live_sized_groups": {
                "size_range": [LIVE_SIZE_MIN, LIVE_SIZE_MAX],
                "groups_per_record_label": GROUPS_PER_RECORD_LABEL,
                **stats,
            },
            "quantile_domain_map": {
                "source": "pooled-label live-sized training rows (benchmark)",
                "target": "100 live-capture chunks (capture_1783985513.json, "
                          "unlabeled calibration)",
                "how": "per-feature monotone ECDF->live-quantile transfer "
                       "baked into TRAINING features; artifact is a bare "
                       "HistGradientBoostingClassifier scoring RAW live "
                       "features, so the miner needs no code change",
                "n_constant_source_features_collapsed": dmap_final.n_collapsed_src,
            },
            "caveat": "if the validator's live regime drifts materially from "
                      "this capture, the map is stale and raw scores may "
                      "drift again; keep the in-batch rank remap hotfix "
                      "deployed as a safety net and re-capture periodically",
        },
        "zero_guard": {
            "fraction": ZERO_ROW_FRACTION,
            "n_zero_rows_final_fit": n_zero_all,
            "p_bot_zero_vector_holdout_model": round(zero_p_holdout, 4),
            "p_bot_zero_vector_final_model": round(zero_p_final, 4),
        },
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_sanitize_errors": n_sanitize_errors,
        "training_dates": uniq,
        "n_examples": int(len(y)),
        "temporal_holdout_date": holdout_date,
        "holdout_metrics": holdout_metrics,
        "live_capture_check": {
            "capture_file": os.path.basename(LIVE_CAPTURE),
            "v2_deployed": live_score_stats(p_live_v2),
            "v2style_raw_holdout_fit": live_score_stats(p_live_raw),
            "v3_holdout_fit": live_score_stats(p_live_v3_holdout),
            "v3_final": live_final_stats,
        },
        "sha256_model_v3_joblib": sha256(model_path),
        "sha256_chunk_features_py": sha256(feat_path),
    }
    meta_path = os.path.join(ARTIFACT_DIR, "model_v3_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {model_path} and {meta_path}")
    return meta


if __name__ == "__main__":
    main()
