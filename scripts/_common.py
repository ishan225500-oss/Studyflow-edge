"""Shared bootstrap for scripts: import paths + results folder."""
import json
import platform
import sys
import sysconfig
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

RESULTS = ROOT / "results"


def machine_info() -> dict:
    info = {
        "python": platform.python_version(),
        "python_platform": sysconfig.get_platform(),   # win-arm64 vs win-amd64 (emulated)
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    return info


def save_result(name: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{name}-{stamp}.json"
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {path.relative_to(ROOT)}")
    return path
