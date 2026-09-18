import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# keep tests away from the real user folder
os.environ.setdefault("STUDYFLOW_DATA_DIR", tempfile.mkdtemp(prefix="studyflow-test-"))
os.environ.setdefault("STUDYFLOW_MODELS_DIR", tempfile.mkdtemp(prefix="studyflow-models-"))
os.environ.setdefault("STUDYFLOW_USE_LLM", "0")
