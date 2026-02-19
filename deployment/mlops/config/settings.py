from __future__ import annotations

from pathlib import Path


class MLOpsSettings:
    ROOT = Path(__file__).resolve().parents[3]
    RESEARCH_DIR = ROOT / "research"
    DEPLOYMENT_DIR = ROOT / "deployment"

    TFRECORDS_DIR = RESEARCH_DIR / "data" / "tfrecords"
    EXPERIMENT = "lung-detection"
    MODEL_STAGE = "Production"

    BATCH_SIZE = 16
    EPOCHS = 20
    VAL_RATIO = 0.2
    MAX_TRAIN_BATCHES = None
    MAX_EVAL_BATCHES = None

