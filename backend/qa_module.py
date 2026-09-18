"""
qa_module.py
Extractive question answering over the user's notes (DistilBERT-SQuAD).

Two interchangeable runtimes share the same tokenisation and answer-span
post-processing:

- "onnx"  : static-shape ONNX model from scripts/export_qa_onnx.py (+ QDQ
            quantisation). Runs on the Snapdragon Hexagon NPU through the
            QNN Execution Provider when available, else on CPU.
- "torch" : the same Hugging Face checkpoint on CPU with PyTorch.

The NPU needs fixed input shapes, so every window is padded to
SETTINGS.qa_seq_len (384) tokens and long passages are split into
overlapping windows (stride 128).
"""
from typing import Dict, List, Optional

import numpy as np

from config import SETTINGS
from utils import build_onnxruntime_session, get_logger, timed

log = get_logger("qa_module")

MAX_ANSWER_TOKENS = 30
STRIDE = 128


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def best_span(start_logits: np.ndarray, end_logits: np.ndarray, context_mask: np.ndarray,
              max_answer_tokens: int = MAX_ANSWER_TOKENS):
    """Pick the (start, end) token pair with the highest start_prob*end_prob,
    restricted to context tokens, end >= start and length <= max_answer_tokens.
    Returns (start, end, score) or None if the window has no context tokens."""
    if not context_mask.any():
        return None
    neg = np.float32(-1e4)
    s = np.where(context_mask, start_logits, neg).astype(np.float64)
    e = np.where(context_mask, end_logits, neg).astype(np.float64)
    ps, pe = _softmax(s), _softmax(e)
    scores = np.triu(np.outer(ps, pe))                       # end >= start
    scores = np.tril(scores, k=max_answer_tokens - 1)        # length limit
    idx = int(np.argmax(scores))
    st, en = divmod(idx, scores.shape[1])
    return st, en, float(scores[st, en])


class QAModule:
    def __init__(self, runtime: Optional[str] = None, model_name: Optional[str] = None,
                 onnx_path=None, prefer_npu: bool = True):
        self.model_name = model_name or SETTINGS.qa_model
        self.requested_runtime = runtime or SETTINGS.qa_runtime
        self.onnx_override = onnx_path
        self.prefer_npu = prefer_npu
        self.seq_len = SETTINGS.qa_seq_len
        self.runtime: Optional[str] = None     # "onnx" | "torch" once loaded
        self.device: str = "not loaded"        # e.g. "NPU (QNNExecutionProvider)"
        self._tokenizer = None
        self._session = None
        self._torch_model = None
        self._input_types: Dict[str, np.dtype] = {}

    # ---------- loading ----------
    def _load(self):
        if self.runtime:
            return
        from transformers import AutoTokenizer

        from pathlib import Path
        onnx_path = Path(self.onnx_override) if self.onnx_override else SETTINGS.qa_onnx_path()
        want = self.requested_runtime
        use_onnx = want == "onnx" or (want == "auto" and onnx_path is not None)
        if want == "onnx" and onnx_path is None:
            raise FileNotFoundError(
                "STUDYFLOW_QA_RUNTIME=onnx but models/qa/model(.qdq).onnx is missing. "
                "Run scripts/export_qa_onnx.py first."
            )

        tok_source = str(onnx_path.parent) if use_onnx and (onnx_path.parent / "tokenizer.json").exists() \
            else self.model_name
        self._tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)

        if use_onnx:
            self._session, provider = build_onnxruntime_session(onnx_path, prefer_npu=self.prefer_npu)
            for inp in self._session.get_inputs():
                self._input_types[inp.name] = np.int32 if "int32" in inp.type else np.int64
                shape = inp.shape
                if len(shape) == 2 and isinstance(shape[1], int):
                    self.seq_len = shape[1]
            self.runtime = "onnx"
            self.device = ("NPU" if provider == "QNNExecutionProvider" else "CPU") + f" ({provider})"
            log.info(f"QA: ONNX {onnx_path.name} on {self.device}, seq_len={self.seq_len}")
        else:
            from transformers import AutoModelForQuestionAnswering
            self._torch_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).eval()
            self.runtime = "torch"
            self.device = "CPU (PyTorch)"
            log.info(f"QA: PyTorch {self.model_name} on CPU")

    def warmup(self):
        self._load()
        self.answer("What is this?", "This is a short warm-up sentence for the model.")

    # ---------- inference ----------
    def _logits(self, input_ids: np.ndarray, attention_mask: np.ndarray):
        if self.runtime == "onnx":
            feeds = {}
            for name, dtype in self._input_types.items():
                src = input_ids if "input_ids" in name else attention_mask
                feeds[name] = src.astype(dtype)
            start, end = self._session.run(None, feeds)[:2]
            return start, end
        import torch
        # inference_mode() must be entered here, not once at load time:
        # torch's grad setting is per-thread, and the model is loaded on the
        # warm-up thread but used from the web server's request threads.
        with torch.inference_mode():
            out = self._torch_model(
                input_ids=torch.from_numpy(input_ids.astype(np.int64)),
                attention_mask=torch.from_numpy(attention_mask.astype(np.int64)),
            )
            return out.start_logits.detach().numpy(), out.end_logits.detach().numpy()

    def _encode(self, question: str, context: str):
        return self._tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.seq_len,
            stride=STRIDE,
            padding="max_length",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )

    @timed
    def answer(self, question: str, context: str, passages: Optional[List[str]] = None) -> Dict:
        """Return {"answer", "score", "found", "context"}. `passages` (from the
        BM25 index) are searched in order; the best-scoring span wins."""
        self._load()
        candidates = passages or [context]
        best = {"answer": "", "score": 0.0, "found": False, "context": ""}
        for passage in candidates:
            if not passage.strip():
                continue
            enc = self._encode(question, passage)
            for w in range(enc["input_ids"].shape[0]):
                ids = enc["input_ids"][w:w + 1]
                mask = enc["attention_mask"][w:w + 1]
                start_logits, end_logits = self._logits(ids, mask)
                seq_ids = enc.sequence_ids(w)
                ctx_mask = np.array([sid == 1 for sid in seq_ids])
                span = best_span(start_logits[0], end_logits[0], ctx_mask)
                if span is None:
                    continue
                st, en, score = span
                if score > best["score"]:
                    offsets = enc["offset_mapping"][w]
                    a, b = int(offsets[st][0]), int(offsets[en][1])
                    best = {
                        "answer": passage[a:b].strip(),
                        "score": score,
                        "found": False,
                        "context": _sentence_around(passage, a, b),
                    }
        best["found"] = bool(best["answer"]) and best["score"] >= SETTINGS.qa_min_score
        best["engine"] = f"DistilBERT QA ({self.device})"
        return best


def _sentence_around(text: str, a: int, b: int, pad: int = 160) -> str:
    dot = text.rfind(". ", 0, a)
    left = max(dot + 2 if dot != -1 else 0, a - pad, 0)
    right_dot = text.find(". ", b)
    right = min(right_dot + 1 if right_dot != -1 else len(text), b + pad)
    return text[left:right].strip()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python backend/qa_module.py <context.txt> \"question\"")
        sys.exit(1)
    ctx = open(sys.argv[1], encoding="utf-8").read()
    qa = QAModule()
    print(qa.answer(sys.argv[2], ctx))
