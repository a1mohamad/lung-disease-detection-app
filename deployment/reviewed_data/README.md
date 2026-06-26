# Reviewed training data

This directory is the handoff point between an assumed clinical review process
and the retraining pipeline. The runtime upload bucket is not training data.

The retraining pipeline accepts either:

- `REVIEWED_DATA_BACKEND=local`: files beneath `reviewed_data/incoming/`
- `REVIEWED_DATA_BACKEND=supabase`: objects in the configured private bucket

Both backends use the same `index.json` and monthly `manifest.json` contract.
Object keys in a manifest are relative to the configured local root or Supabase
prefix. Every sample requires a reviewed image, lung mask, class label, and
patient identifier.

The checked-in `examples/` files document the contract only. They are not read
by the pipeline. Put real handoff files under the ignored `incoming/` directory
or in the configured Supabase bucket.

The default `legacy` mode keeps using the existing TFRecords exactly as before.
Set `RETRAIN_DATASET_MODE=prepared` only when reviewed data is available.

Recommended rollout:

1. Keep `RETRAIN_DRY_RUN=true` while validating manifests, snapshots, metrics,
   Keras export, and ONNX parity. This cannot move the MLflow production alias
   or upload to Hugging Face.
2. Configure `HF_PUBLISH_REPO_ID` and a write-scoped `HF_TOKEN`, then set
   `HF_PUBLISH_ENABLED=true`.
3. Set `RETRAIN_DRY_RUN=false` only when automatic production delivery is
   intended. The candidate is uploaded to Hugging Face first; the MLflow
   `production` alias moves only after that upload succeeds.

`HF_PUBLISH_CREATE_PR=false` is the automatic production-sync mode. A pull
request is useful for manual review, but it does not update the revision used by
the runtime until somebody merges it.
