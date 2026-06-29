import os
import tempfile
from pathlib import Path


# Keep test imports isolated from external infrastructure.
os.environ["DB_LOGGING_ENABLED"] = "false"
os.environ["KAFKA_ENABLED"] = "false"
os.environ["MLFLOW_ENABLED"] = "false"
os.environ["HF_MODEL_DOWNLOAD_ENABLED"] = "false"

# VS Code/Windows can lock pytest temp folders between runs. Keep pytest's
# tmp_path root inside the project, but let pytest create numbered run folders.
_TMP_ROOT = Path(__file__).resolve().parents[1] / ".pytest-tmp-root"
_TMP_ROOT.mkdir(exist_ok=True)
os.environ["TEMP"] = str(_TMP_ROOT)
os.environ["TMP"] = str(_TMP_ROOT)
os.environ["TMPDIR"] = str(_TMP_ROOT)
tempfile.tempdir = str(_TMP_ROOT)
