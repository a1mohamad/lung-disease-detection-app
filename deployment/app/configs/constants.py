from __future__ import annotations

import numpy as np


IMAGENET_TORCH_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_TORCH_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

ONNX_EXPORT_DEFAULT_OPSET = 13
ONNX_VALIDATION_RANDOM_SEED = 42
ONNX_VALIDATION_DEFAULT_RTOL = 1e-3
ONNX_VALIDATION_DEFAULT_ATOL = 1e-3
