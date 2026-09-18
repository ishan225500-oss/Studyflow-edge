"""
utils.py
Shared helpers: logging, timing, text chunking and the ONNX Runtime
session builder that prefers the Snapdragon Hexagon NPU (QNN EP).
"""
import functools
import logging
import re
import time
from typing import List

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def timed(fn):
    """Log wall-clock latency of a pipeline stage."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        get_logger(fn.__module__).info(
            f"{fn.__qualname__} took {(time.perf_counter() - start) * 1000:.1f} ms"
        )
        return result
    return wrapper


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", flat) if s.strip()]


def chunk_text(text: str, max_chars: int = 2000) -> List[str]:
    """Split text into chunks on paragraph/sentence boundaries. A paragraph
    longer than max_chars is split by sentences, and a sentence longer than
    max_chars is hard-split."""
    units: List[str] = []
    for para in (p.strip() for p in text.split("\n") if p.strip()):
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in split_sentences(para):
            while len(sent) > max_chars:
                units.append(sent[:max_chars])
                sent = sent[max_chars:]
            if sent:
                units.append(sent)

    chunks, current = [], ""
    for u in units:
        if current and len(current) + len(u) + 1 > max_chars:
            chunks.append(current)
            current = u
        else:
            current = f"{current}\n{u}" if current else u
    if current:
        chunks.append(current)
    return chunks


def build_onnxruntime_session(model_path, prefer_npu: bool = True, npu_only: bool = False):
    """
    Build an ONNX Runtime InferenceSession.

    On a Snapdragon-powered HP PC with `onnxruntime-qnn` installed, the model
    runs on the Hexagon NPU through the QNN Execution Provider. Anywhere else
    it falls back to the CPU so the app still works.

    npu_only=True disables CPU fallback for unsupported ops, which is how
    scripts/check_npu.py proves the whole graph is offloaded to the NPU.
    Returns (session, provider_actually_used).
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = []
    so = ort.SessionOptions()

    if prefer_npu and "QNNExecutionProvider" in available:
        providers.append((
            "QNNExecutionProvider",
            {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_graph_finalization_optimization_mode": "3",
            },
        ))
        if npu_only:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    elif npu_only:
        raise RuntimeError(
            "QNNExecutionProvider is not available. Install onnxruntime-qnn on a "
            f"Snapdragon Windows PC. Available providers: {available}"
        )

    if not npu_only:
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
    return session, session.get_providers()[0]
