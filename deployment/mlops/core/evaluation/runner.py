from __future__ import annotations

from typing import Any

from mlops.core.data.datasets import (
    build_binary_dataset,
    build_multiclass_dataset,
    build_segmentation_dataset,
)
from mlops.core.evaluation.metrics import eval_binary, eval_multiclass, eval_segmentation
from mlops.core.models.compile import get_preprocess_fn


def build_eval_dataset_for_spec(spec, metadata: dict[str, Any], val_files, batch_size: int):
    image_size = tuple(metadata.get("inference", {}).get("input_size", [256, 256]))
    preprocess_config = metadata.get("preprocessing", {})

    if spec.task == "binary_classification":
        prep = preprocess_config.get("preprocess_input_fn", "")
        preprocess_fn = get_preprocess_fn(prep)
        return build_binary_dataset(val_files, image_size, batch_size, preprocess_fn, preprocess_config)

    if spec.task == "multiclass_classification":
        prep = preprocess_config.get("preprocess_input_fn", "")
        preprocess_fn = get_preprocess_fn(prep)
        num_classes = len(metadata.get("output", {}).get("classes", {}))
        return build_multiclass_dataset(
            val_files,
            image_size,
            batch_size,
            preprocess_fn,
            num_classes,
            preprocess_config,
        )

    return build_segmentation_dataset(val_files, image_size, batch_size, preprocess_config)


def evaluate_model_for_spec(spec, model, dataset, max_eval_batches: int | None, metadata: dict[str, Any]):
    if spec.task == "binary_classification":
        return eval_binary(model, dataset, max_eval_batches)
    if spec.task == "multiclass_classification":
        num_classes = len(metadata.get("output", {}).get("classes", {}))
        return eval_multiclass(model, dataset, max_eval_batches, num_classes)
    return eval_segmentation(model, dataset, max_eval_batches)

