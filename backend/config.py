"""
config.py
Runtime settings, all overridable with environment variables.
Importing this module also points the Hugging Face cache at models_dir()
so that weights fetched by scripts/download_models.py are found offline
and can be bundled into the desktop build.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import paths


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def configure_hf_cache() -> Path:
    hf_home = paths.models_dir() / "hf"
    os.environ.setdefault("HF_HOME", str(hf_home))
    # If weights are already on disk, never touch the network.
    hub = hf_home / "hub"
    if hub.is_dir() and any(hub.iterdir()):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return hf_home


configure_hf_cache()


@dataclass
class Settings:
    # OCR: "easyocr" (printed) or "handwriting" (EasyOCR line detection + TrOCR handwritten)
    ocr_backend: str = os.environ.get("STUDYFLOW_OCR_BACKEND", "easyocr")
    ocr_languages: list = field(
        default_factory=lambda: os.environ.get("STUDYFLOW_OCR_LANGS", "en").split(",")
    )
    trocr_model: str = os.environ.get("STUDYFLOW_TROCR_MODEL", "microsoft/trocr-small-handwritten")

    # Classic (non-LLM) fallback models
    sum_model: str = os.environ.get("STUDYFLOW_SUM_MODEL", "t5-small")
    qg_model: str = os.environ.get("STUDYFLOW_QG_MODEL", "valhalla/t5-small-qg-hl")
    qa_model: str = os.environ.get("STUDYFLOW_QA_MODEL", "distilbert-base-uncased-distilled-squad")

    # QA runtime: "auto" uses the ONNX model (NPU via QNN if present) when
    # models/qa/ has one, otherwise PyTorch on CPU. "onnx" / "torch" force one.
    qa_runtime: str = os.environ.get("STUDYFLOW_QA_RUNTIME", "auto")
    qa_seq_len: int = int(os.environ.get("STUDYFLOW_QA_SEQ_LEN", "384"))
    qa_min_score: float = float(os.environ.get("STUDYFLOW_QA_MIN_SCORE", "0.15"))

    # Local LLM (llama.cpp GGUF). Empty path = auto-detect models/llm/*.gguf
    use_llm: bool = _env_bool("STUDYFLOW_USE_LLM", True)
    llm_path: str = os.environ.get("STUDYFLOW_LLM_GGUF", "")
    llm_ctx: int = int(os.environ.get("STUDYFLOW_LLM_CTX", "4096"))
    llm_threads: int = int(os.environ.get("STUDYFLOW_LLM_THREADS", "0"))  # 0 = auto

    keep_uploads: bool = _env_bool("STUDYFLOW_KEEP_UPLOADS", False)
    max_upload_mb: int = int(os.environ.get("STUDYFLOW_MAX_UPLOAD_MB", "50"))

    # Work limits: file size alone doesn't bound OCR/render cost.
    max_pdf_pages: int = int(os.environ.get("STUDYFLOW_MAX_PDF_PAGES", "300"))
    max_ocr_pages: int = int(os.environ.get("STUDYFLOW_MAX_OCR_PAGES", "40"))
    max_image_mp: float = float(os.environ.get("STUDYFLOW_MAX_IMAGE_MP", "20"))   # megapixels fed to OCR
    max_text_chars: int = int(os.environ.get("STUDYFLOW_MAX_TEXT_CHARS", "3000000"))

    def qa_onnx_path(self) -> Optional[Path]:
        d = paths.models_dir() / "qa"
        for name in ("model.qdq.onnx", "model.onnx"):
            if (d / name).is_file():
                return d / name
        return None

    def llm_gguf_path(self) -> Optional[Path]:
        if self.llm_path:
            p = Path(self.llm_path)
            return p if p.is_file() else None
        d = paths.models_dir() / "llm"
        if d.is_dir():
            ggufs = sorted(d.glob("*.gguf"))
            if ggufs:
                return ggufs[0]
        return None


SETTINGS = Settings()
