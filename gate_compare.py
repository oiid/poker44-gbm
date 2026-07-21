#!/usr/bin/env python
"""Honest paired-window deploy gate for Poker44 (SN126) artifacts.

Compares a CANDIDATE artifact against an INCUMBENT artifact purely on the
metric the validator pays for: exact poker44.score.scoring.reward computed
per simulated live request window, with each side scored through its EXACT
serving path (its feature module + its serving remap).  There is NO
live-spread constraint: under the rank-preserving serving remap only the
within-batch ordering matters, so raw-score spread is an invalid proxy
(the Jul-16 selection incident: the trio with the best paired-window reward
was rejected on spread).

Windows: fresh live-sized labeled groups (80-100 sanitized hands) built
from the NEWEST --n-dates benchmark dates (new releases are downloaded
first), sampled into --n-windows batches of --window-size with a 50/50
bot/human mix, identical window indices for both sides (paired).
IMPORTANT — memorization (measured 2026-07-19): a model must never be
gated on windows built from dates it trained on.  model_prev (trained
through 07-18) beat the stronger model_ens 29-30/30 on 07-18+07-19
windows, yet LOST 20/30 on 07-19-only windows (unseen by both).  Salting
the group composition (done here, salt "gatecmp") kills byte-identical
row reuse but NOT hand-level memorization — the 30/30 sweep persisted on
salted 07-18 windows.  Callers must pick --n-dates so the gate dates are
unseen by BOTH sides (nightly: gate a twin trained with
POKER44_EXCLUDE_NEWEST=2).  This script warns when an adjacent
*_meta.json shows a side trained on a gate date.

Serving paths:
  features v1 -> neurons.chunk_features (111 cols, deployed extractor)
  features v3 -> poker44-staging chunk_features_v3 (468 cols; bit-identical
                 to the post-boundary neurons/chunk_features_v3)
  remap deployed -> the real neurons/model_miner.py Miner._remap_in_batch
  remap staged   -> the real poker44-staging/model_miner.py Miner._remap_in_batch
                    (imported through the deploysim package, artifact-driven
                    positive fraction)
  remap auto (default) -> staged when that side uses v3 features, else
                          deployed.  Post-boundary the two are the same code.
Rank-branch ensemble members rank WITHIN the incoming batch, so
predict_proba is called per window (exactly like one live request), never
pooled.

Verdict DEPLOY requires ALL of:
  1. candidate wins strictly more reward on >= --win-threshold (default 60%)
     of the windows;
  2. mean paired reward delta > 0;
  3. candidate P(bot | zero feature vector) < 0.1 (failed-extraction guard).
Exit code 0 = DEPLOY, 1 = KEEP incumbent, 2 = error.

Examples:
  # nightly gate (both sides currently v1 features, deployed remap):
  PYTHONPATH=/root/bittensor/Poker44-subnet miner_env/bin/python \
    gate_compare.py --candidate model_artifacts/model_v3.joblib \
                    --incumbent model_artifacts/model.joblib
  # boundary gate (staged v4 vs deployed):
  ... gate_compare.py --candidate poker44-staging/model_v4.joblib \
        --features-candidate v3 --incumbent model_artifacts/model.joblib

Caveat (nightly use): a candidate retrained through the newest date has
seen the gate dates' hands in training; the comparison is still paired and
serving-exact, but treat wins near the threshold with suspicion.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import inspect
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/bittensor/Poker44-subnet"
STAGING = "/root/bittensor/poker44-staging"
CACHE_DIR = os.path.join(DATA_DIR, "gate_cache")

sys.path.insert(0, REPO)
sys.path.insert(0, DATA_DIR)
sys.path.insert(0, STAGING)

WIN_THRESHOLD = 0.60
ZERO_P_MAX = 0.10


# ---------------------------------------------------------------------------
# data: newest-N-dates live-sized labeled groups (cached)
# ---------------------------------------------------------------------------

def ensure_downloads():
    """Fetch any new benchmark releases before building windows."""
    r = subprocess.run([sys.executable, os.path.join(DATA_DIR, "download.py")],
                      capture_output=True, text=True,
                      env={**os.environ, "PYTHONPATH": REPO})
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    print(f"download.py rc={r.returncode}: {tail}")
    if r.returncode != 0:
        print(f"WARNING: download failed ({(r.stderr or '')[-300:]}); "
              "continuing with existing files")


def newest_dates(n):
    dates = sorted(os.path.basename(p)[4:-5]
                   for p in glob.glob(os.path.join(DATA_DIR, "raw_*.json")))
    if len(dates) < n:
        sys.exit(f"only {len(dates)} raw files present, need {n}")
    return dates[-n:]


GROUP_SALT = "gatecmp"   # NEVER "live_sized": see module docstring


def load_groups(dates):
    """Live-sized labeled hand groups for the given dates, composed with the
    training recipe but a gate-specific seed salt (anti-memorization).
    Cached."""
    import train_v3 as tv3

    os.makedirs(CACHE_DIR, exist_ok=True)
    key = hashlib.sha256(("|".join(dates) + "::" + GROUP_SALT).encode()) \
        .hexdigest()[:16]
    cache = os.path.join(CACHE_DIR, f"groups_{key}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            payload = pickle.load(f)
        if payload["dates"] == list(dates):
            print(f"groups cache hit: {cache} ({len(payload['y'])} groups)")
            return payload["groups"], np.asarray(payload["y"]), payload["dates"]

    records, errors = [], []
    for date in dates:
        path = os.path.join(DATA_DIR, f"raw_{date}.json")
        with open(path) as f:
            recs = json.load(f)
        for rec in recs:
            rec_id = str(rec.get("chunkId", ""))
            pairs = []
            for sub, gt in zip(rec.get("chunks", []), rec.get("groundTruth", [])):
                pairs.append((int(gt), tv3.sanitize_subchunk(sub, errors)))
            if pairs:
                records.append((date, rec_id, pairs))
        print(f"sanitized {date}: {len(recs)} records")
    if errors:
        print(f"WARNING: sanitizer raised on {len(errors)} hands")

    by_date_label = {}
    for date, rec_id, pairs in records:
        for label, sub in pairs:
            by_date_label.setdefault((date, label), []).append((rec_id, sub))
    groups, y = [], []
    for date, rec_id, pairs in records:
        for label in (0, 1):
            subs = [sub for lbl, sub in pairs if lbl == label]
            if not subs:
                continue
            borrow = [s for rid, s in by_date_label.get((date, label), [])
                      if rid != rec_id]
            for k in range(tv3.GROUPS_PER_RECORD_LABEL):
                rng = tv3._rng_for(rec_id, label, GROUP_SALT, k)
                target = rng.randint(tv3.LIVE_SIZE_MIN, tv3.LIVE_SIZE_MAX)
                hands, _ = tv3.build_group(subs, target, rng, borrow)
                if hands is None:
                    continue
                groups.append(hands)
                y.append(label)
    y = np.asarray(y, dtype=int)
    print(f"built {len(groups)} live-sized groups "
          f"({int(y.sum())} bot / {int((y == 0).sum())} human) from {dates}")
    with open(cache, "wb") as f:
        pickle.dump({"dates": list(dates), "groups": groups,
                     "y": y.tolist()}, f)
    return groups, y, list(dates)


def featurize(groups, version, dates):
    """Feature matrix through the given serving feature module.  Cached."""
    key = hashlib.sha256(
        ("|".join(dates) + f"::{version}::{GROUP_SALT}").encode()) \
        .hexdigest()[:16]
    cache = os.path.join(CACHE_DIR, f"feats_{version}_{key}.npz")
    if os.path.exists(cache):
        X = np.load(cache)["X"]
        if len(X) == len(groups):
            print(f"feature cache hit: {cache} {X.shape}")
            return X
    if version == "v1":
        from neurons.chunk_features import extract_features
    elif version == "v3":
        from chunk_features_v3 import extract_features
    else:
        sys.exit(f"unknown feature version {version!r}")
    t0 = time.time()
    X = np.asarray([extract_features(g) for g in groups], dtype=float)
    print(f"extracted {version} features {X.shape} in {time.time()-t0:.0f}s")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, X=X)
    return X


# ---------------------------------------------------------------------------
# serving remaps: imported from the REAL miner implementations
# ---------------------------------------------------------------------------

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _adapt_remap(miner_cls, model):
    """Return remap(scores, chunks) -> list using the class's real method,
    whatever its era: classmethod(scores) [deployed pre-boundary] or
    instance method (scores, chunks) with artifact-driven fraction [staged]."""
    static = inspect.getattr_static(miner_cls, "_remap_in_batch")
    if isinstance(static, (classmethod, staticmethod)):
        fn = getattr(miner_cls, "_remap_in_batch")
        return lambda scores, chunks: fn(list(scores))
    inst = miner_cls.__new__(miner_cls)   # skip neuron __init__
    inst.model = model
    params = list(inspect.signature(static).parameters)  # self, scores[, chunks]
    if "chunks" in params:
        return lambda scores, chunks: inst._remap_in_batch(list(scores), chunks)
    return lambda scores, chunks: inst._remap_in_batch(list(scores))


_STAGED_SIM_READY = False


def _ensure_deploysim():
    """(Re)build the deploysim symlink package (staging-only, like verify_v4)."""
    global _STAGED_SIM_READY
    sim_neurons = os.path.join(STAGING, "deploysim", "neurons")
    os.makedirs(sim_neurons, exist_ok=True)
    links = {
        "__init__.py": os.path.join(REPO, "neurons", "__init__.py"),
        "chunk_features.py": os.path.join(REPO, "neurons", "chunk_features.py"),
        "miner.py": os.path.join(REPO, "neurons", "miner.py"),
        "chunk_features_v2.py": os.path.join(STAGING, "chunk_features_v2.py"),
        "chunk_features_v3.py": os.path.join(STAGING, "chunk_features_v3.py"),
        "model_miner.py": os.path.join(STAGING, "model_miner.py"),
    }
    for name, target in links.items():
        link = os.path.join(sim_neurons, name)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
    _STAGED_SIM_READY = True


_MODULE_CACHE = {}


def _serving_module(kind):
    if kind in _MODULE_CACHE:
        return _MODULE_CACHE[kind]
    if kind == "deployed":
        mod = _load_module("gate_serving_deployed_mm",
                           os.path.join(REPO, "neurons", "model_miner.py"))
    elif kind == "staged":
        _ensure_deploysim()
        sim = os.path.join(STAGING, "deploysim")
        # staged model_miner imports `neurons.chunk_features_v3`; resolve the
        # neurons package to the deploysim tree for the import, then restore.
        saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                      if k == "neurons" or k.startswith("neurons.")}
        sys.path.insert(0, sim)
        try:
            mod = _load_module("gate_serving_staged_mm",
                               os.path.join(sim, "neurons", "model_miner.py"))
        finally:
            sys.path.remove(sim)
            for k in [k for k in list(sys.modules)
                      if k == "neurons" or k.startswith("neurons.")]:
                del sys.modules[k]
            sys.modules.update(saved_mods)
    else:
        sys.exit(f"unknown remap kind {kind!r}")
    _MODULE_CACHE[kind] = mod
    return mod


def get_remap(kind, model):
    """kind in {deployed, staged} -> remap(scores, chunks) via the real code."""
    return _adapt_remap(_serving_module(kind).Miner, model)


def staged_tie_keys(groups):
    """Real serving tie key per group, computed once.  Evaluation loops can
    pass these strings as the `chunks` argument of a staged remap: hashing a
    short key string keeps the remap deterministic and order-invariant while
    avoiding re-hashing ~100 KB of hand JSON per chunk per window (only the
    order of exact float ties can differ from serving; remapped band values
    for ties are within one band step, immaterial to the reward)."""
    miner_cls = _serving_module("staged").Miner
    return [miner_cls._chunk_tie_key(g) for g in groups]


# ---------------------------------------------------------------------------
# paired-window scoring
# ---------------------------------------------------------------------------

def make_windows(y, n_windows, size, seed, bot_frac=0.5):
    rng = np.random.RandomState(seed)
    bots = np.flatnonzero(y == 1)
    hums = np.flatnonzero(y == 0)
    n_bot = int(round(size * bot_frac))
    n_hum = size - n_bot
    if len(bots) < n_bot or len(hums) < n_hum:
        sys.exit(f"not enough groups for windows ({len(bots)} bot/"
                 f"{len(hums)} human, need {n_bot}/{n_hum})")
    windows = []
    for _ in range(n_windows):
        idx = np.concatenate([rng.choice(bots, size=n_bot, replace=False),
                              rng.choice(hums, size=n_hum, replace=False)])
        rng.shuffle(idx)
        windows.append(idx)
    return windows


def score_side(model, X, groups, y, windows, remap_fn, tag=""):
    """Per-window serving-exact rewards: predict_proba PER WINDOW (rank
    members recalibrate within the batch, exactly like one live request),
    then the side's real remap, then the exact reward."""
    from poker44.score.scoring import reward

    rewards = []
    t0 = time.time()
    for idx in windows:
        p = model.predict_proba(X[idx])[:, 1]
        p = [max(0.0, min(1.0, float(v))) for v in p]        # serving clamp
        remapped = remap_fn(p, [groups[i] for i in idx])
        r, _ = reward(np.asarray(remapped, dtype=float), y[idx])
        rewards.append(float(r))
    print(f"  scored {len(windows)} windows{(' [' + tag + ']') if tag else ''} "
          f"in {time.time()-t0:.0f}s")
    return np.asarray(rewards)


def window_summary(r):
    return {"mean": round(float(r.mean()), 4), "std": round(float(r.std()), 4),
            "robust_mean_minus_half_std": round(float(r.mean() - 0.5 * r.std()), 4),
            "p10": round(float(np.percentile(r, 10)), 4),
            "min": round(float(r.min()), 4),
            "n_zero": int((r == 0).sum())}


def zero_vector_p(model, n_features):
    return float(model.predict_proba(np.zeros((1, n_features)))[0, 1])


def contamination_check(artifact_path, gate_dates, side):
    """Warn when the artifact's adjacent meta shows it trained on a gate
    date (hand-level memorization invalidates the paired comparison)."""
    meta_path = artifact_path.rsplit(".joblib", 1)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        print(f"note: no meta next to {os.path.basename(artifact_path)}; "
              f"cannot check {side} for gate-date contamination")
        return None
    try:
        with open(meta_path) as f:
            trained = set(json.load(f).get("training_dates") or [])
    except Exception as exc:  # noqa: BLE001
        print(f"note: unreadable meta {meta_path}: {exc!r}")
        return None
    overlap = sorted(trained & set(gate_dates))
    if overlap:
        print(f"WARNING: {side} trained on gate date(s) {overlap} — "
              "memorization will inflate its windows there; the verdict "
              "is NOT trustworthy in its favor on those dates")
    return overlap


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def compare(candidate_path, incumbent_path, feat_cand="v1", feat_inc="v1",
            remap_cand="auto", remap_inc="auto", n_dates=2, n_windows=30,
            window_size=100, seed=7, bot_frac=0.5, download=True):
    import joblib

    if download:
        ensure_downloads()
    dates = newest_dates(n_dates)
    print(f"gate dates: {dates}")
    contam_c = contamination_check(candidate_path, dates, "CANDIDATE")
    contam_i = contamination_check(incumbent_path, dates, "INCUMBENT")
    groups, y, dates = load_groups(dates)

    X_c = featurize(groups, feat_cand, dates)
    X_i = X_c if feat_inc == feat_cand else featurize(groups, feat_inc, dates)

    t0 = time.time()
    cand = joblib.load(candidate_path)
    inc = joblib.load(incumbent_path)
    print(f"loaded artifacts in {time.time()-t0:.1f}s: "
          f"candidate={os.path.basename(candidate_path)} "
          f"({type(cand).__name__}), incumbent="
          f"{os.path.basename(incumbent_path)} ({type(inc).__name__})")

    rk_c = ("staged" if feat_cand == "v3" else "deployed") \
        if remap_cand == "auto" else remap_cand
    rk_i = ("staged" if feat_inc == "v3" else "deployed") \
        if remap_inc == "auto" else remap_inc
    print(f"serving paths: candidate = features {feat_cand} + {rk_c} remap; "
          f"incumbent = features {feat_inc} + {rk_i} remap")
    remap_c = get_remap(rk_c, cand)
    remap_i = get_remap(rk_i, inc)

    windows = make_windows(y, n_windows, window_size, seed, bot_frac)
    r_c = score_side(cand, X_c, groups, y, windows, remap_c, "candidate")
    r_i = score_side(inc, X_i, groups, y, windows, remap_i, "incumbent")

    delta = r_c - r_i
    wins = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    losses = int((delta < 0).sum())
    win_rate = wins / len(delta)
    mean_delta = float(delta.mean())
    p0_c = zero_vector_p(cand, X_c.shape[1])
    p0_i = zero_vector_p(inc, X_i.shape[1])

    deploy = (win_rate >= WIN_THRESHOLD and mean_delta > 0
              and p0_c < ZERO_P_MAX)
    result = {
        "candidate": candidate_path, "incumbent": incumbent_path,
        "features": {"candidate": feat_cand, "incumbent": feat_inc},
        "remap": {"candidate": rk_c, "incumbent": rk_i},
        "gate_dates": dates,
        "gate_date_contamination": {"candidate": contam_c,
                                    "incumbent": contam_i},
        "n_groups": int(len(y)), "n_bot": int(y.sum()),
        "windows": {"n": n_windows, "size": window_size, "seed": seed,
                    "bot_frac": bot_frac},
        "candidate_windows": window_summary(r_c),
        "incumbent_windows": window_summary(r_i),
        "paired": {"wins": wins, "ties": ties, "losses": losses,
                   "win_rate": round(win_rate, 4),
                   "mean_delta": round(mean_delta, 4),
                   "delta_std": round(float(delta.std()), 4)},
        "p_bot_zero_vector": {"candidate": round(p0_c, 4),
                              "incumbent": round(p0_i, 4)},
        "criteria": {
            "win_rate_ge_0.60": win_rate >= WIN_THRESHOLD,
            "mean_delta_gt_0": mean_delta > 0,
            "candidate_zero_p_lt_0.1": p0_c < ZERO_P_MAX,
        },
        "verdict": "DEPLOY" if deploy else "KEEP_INCUMBENT",
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--features-candidate", choices=("v1", "v3"), default="v1")
    ap.add_argument("--features-incumbent", choices=("v1", "v3"), default="v1")
    ap.add_argument("--remap-candidate", choices=("auto", "deployed", "staged"),
                    default="auto")
    ap.add_argument("--remap-incumbent", choices=("auto", "deployed", "staged"),
                    default="auto")
    ap.add_argument("--n-dates", type=int, default=2)
    ap.add_argument("--n-windows", type=int, default=30)
    ap.add_argument("--window-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--bot-frac", type=float, default=0.5)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    result = compare(
        args.candidate, args.incumbent,
        feat_cand=args.features_candidate, feat_inc=args.features_incumbent,
        remap_cand=args.remap_candidate, remap_inc=args.remap_incumbent,
        n_dates=args.n_dates, n_windows=args.n_windows,
        window_size=args.window_size, seed=args.seed, bot_frac=args.bot_frac,
        download=not args.no_download)

    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
    print(f"\nVERDICT: {result['verdict']} "
          f"(wins {result['paired']['wins']}/{result['windows']['n']}, "
          f"mean delta {result['paired']['mean_delta']:+.4f}, "
          f"candidate P(bot|zeros) {result['p_bot_zero_vector']['candidate']})")
    sys.exit(0 if result["verdict"] == "DEPLOY" else 1)


if __name__ == "__main__":
    main()
