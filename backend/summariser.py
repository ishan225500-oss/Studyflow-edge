"""
summariser.py
Fallback summariser used when no local LLM is installed: T5-small (or any
seq2seq checkpoint) run directly with transformers on CPU.

Uses the model + tokenizer directly (not pipeline()) so the "summarize:"
prefix is applied exactly once and behaviour doesn't change across
transformers versions.
"""
from typing import List, Optional

from config import SETTINGS
from utils import chunk_text, get_logger, timed

log = get_logger("summariser")

MAX_CHUNKS = 8   # evenly sampled across long documents to bound latency


def _spread(items: List[str], k: int) -> List[str]:
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


class Summarizer:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or SETTINGS.sum_model
        self._tok = None
        self._model = None

    @property
    def prefix(self) -> str:
        return "summarize: " if "t5" in self.model_name.lower() else ""

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            log.info(f"Loading summariser: {self.model_name}")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
        return self._tok, self._model

    def warmup(self):
        self._summarize_one("Warm-up text for the summarisation model. " * 5, 20, 5)

    def _summarize_one(self, text: str, max_new: int, min_new: int) -> str:
        tok, model = self._load()
        inputs = tok(self.prefix + text, return_tensors="pt", truncation=True, max_length=512)
        import torch
        with torch.inference_mode():
            ids = model.generate(
                **inputs, max_new_tokens=max_new, min_new_tokens=min_new,
                num_beams=4, no_repeat_ngram_size=3, early_stopping=True,
            )
        return tok.decode(ids[0], skip_special_tokens=True).strip()

    @timed
    def summarize(self, text: str, max_new: int = 150) -> str:
        if not text.strip():
            return ""
        chunks = _spread(chunk_text(text, max_chars=1800), MAX_CHUNKS)
        parts = []
        for c in chunks:
            # don't force long outputs from short chunks (causes padding/hallucination)
            min_new = min(30, max(5, len(c.split()) // 6))
            parts.append(self._summarize_one(c, max_new, min_new))
        if len(parts) == 1:
            return parts[0]
        # present per-section points instead of re-summarising (which truncates)
        return "\n".join(f"- {p}" for p in parts if p)
