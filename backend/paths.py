"""
paths.py
One place that decides where things live, so the same code works when
run from source and when frozen into StudyFlowEdge.exe by PyInstaller.

- resource_dir(): read-only files shipped with the app (templates, bundled models)
- user_data_dir(): writable per-user folder (%LOCALAPPDATA%\\StudyFlowEdge on Windows)
- models_dir(): where model weights are loaded from / downloaded to
"""
import os
import sys
from pathlib import Path

APP_NAME = "StudyFlowEdge"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False)) and hasattr(sys, "_MEIPASS")


def resource_dir() -> Path:
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    override = os.environ.get("STUDYFLOW_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_NAME
    else:
        base = Path.home() / ".studyflow-edge"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _has_files(p: Path) -> bool:
    return p.is_dir() and any(x.name != ".gitkeep" for x in p.iterdir())


def models_dir() -> Path:
    override = os.environ.get("STUDYFLOW_MODELS_DIR")
    if override:
        p = Path(override)
    elif is_frozen():
        bundled = resource_dir() / "models"
        p = bundled if _has_files(bundled) else user_data_dir() / "models"
    else:
        p = resource_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def uploads_dir() -> Path:
    p = user_data_dir() / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def templates_dir() -> Path:
    return resource_dir() / "app" / "templates"
