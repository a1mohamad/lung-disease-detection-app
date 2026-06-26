from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tensorflow as tf

from mlops.core.publishing.release import _convert_saved_model, _validate_onnx


class OnnxReleaseSmokeTests(unittest.TestCase):
    def test_saved_model_conversion_matches_tensorflow(self):
        import onnxruntime as ort

        inputs = tf.keras.Input(shape=(4,), name="features")
        outputs = tf.keras.layers.Dense(
            2,
            use_bias=True,
            kernel_initializer="ones",
            bias_initializer="zeros",
        )(inputs)
        model = tf.keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            saved_model_dir = root / "saved_model"
            onnx_path = root / "model.onnx"
            model.export(str(saved_model_dir))
            _convert_saved_model(saved_model_dir, onnx_path)
            result = _validate_onnx(
                model,
                onnx_path,
                [None, 4],
                ort,
            )

        self.assertTrue(result["valid"])


if __name__ == "__main__":
    unittest.main()
