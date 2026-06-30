"""Dataset construction and TFRecord utilities."""

# Dataset builders and TFRecord utilities are re-exported for task code that
# needs a compact MLOps data API.
from mlops.core.data.datasets import (
    build_binary_dataset,
    build_multiclass_dataset,
    build_segmentation_dataset,
)
from mlops.core.data.tfrecord_ops import (
    compute_steps_from_tfrecords,
    count_examples_in_tfrecords,
    list_tfrecords,
    split_tfrecords,
)
