# poker44-gbm

Miner for Bittensor subnet 126 (Poker44 — poker bot detection). UID 226.

## Approach

Chunk-level bot detection with a HistGradientBoostingClassifier over 111
behavioral features (action-type shares per street, bet-sizing statistics in
big-blind units, sizing-distinctness ratios, street-reach rates, hero-seat
aggregates). See `chunk_features.py` for the exact feature definitions.

Current production model (`train_v4.py`): a seed-averaged soft blend of a
five-family stacked ensemble (LightGBM, XGBoost, CatBoost, ExtraTrees,
RandomForest with a logistic meta-learner) and a within-request
rank-transformed HistGradientBoosting branch, selected by simulated
request-window reward. Served artifact hash-pinned in `model_meta.json`.

Base pipeline (`train_v3.py`, superseded by `train_v4.py` but kept as fallback):

1. Download the public Poker44 training benchmark
   (`https://api.poker44.net/api/v1/benchmark`, daily labeled releases).
2. Pass every hand through the validator's own sanitizer
   (`poker44.validator.payload_view.prepare_hand_for_miner`) so training
   matches the serve-time payload exactly.
3. Build training groups at live chunk sizes (80–100 hands) by concatenating
   same-label player-session sub-chunks.
4. Apply a rank-preserving per-feature quantile domain map from benchmark
   marginals to observed live-payload marginals.
5. Train HGB with a ~1% zero-feature-vector guard labeled human.

Serving (`model_miner.py`): scores each received chunk with the model, then
applies an in-batch rank-preserving remap so a bounded fraction of chunks per
request crosses the 0.5 flag threshold.

## Training data statement

Trained exclusively on the public Poker44 training benchmark releases and the
miner-visible live payload structure. No validator-only evaluation data, no
private data.

## Artifact

The served model artifact is hash-pinned in `model_v3_meta.json`
(`sha256_model_v3_joblib`); metrics on the temporal holdout are recorded there.

## License

MIT
