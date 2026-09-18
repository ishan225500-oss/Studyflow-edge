"""
engine.py
StudyEngine ties the pipeline together and picks the best available
backend for each task:

    file -> TextExtractor -> BM25 index -> summary + practice questions
    question -> BM25 top passages -> DistilBERT QA (answer + quoted line)
             -> LLM explanation of that answer (if installed)

Every result says which engine/device produced it, so the UI and the
benchmarks report honestly what ran where.
"""
import threading
import time
from typing import Callable, Dict, List, Optional

import llm as llm_mod
from config import SETTINGS
from qa_module import QAModule
from question_gen import QuestionGenerator
from retrieval import BM25Index
from summariser import Summarizer
from text_extract import TextExtractor
from utils import get_logger

log = get_logger("engine")

Progress = Optional[Callable[[str, str], None]]   # (stage, message)


class StudyEngine:
    def __init__(self, extractor=None, summarizer=None, qgen=None, qa=None, local_llm=None):
        self.extractor = extractor or TextExtractor()
        self.summarizer = summarizer or Summarizer()
        self.qgen = qgen or QuestionGenerator()
        self.qa = qa or QAModule()
        if local_llm is not None:
            self.llm = local_llm
        else:
            self.llm = llm_mod.LocalLLM() if llm_mod.available() else None
        self._lock = threading.RLock()
        self.warm = False
        self.warmup_error: Optional[str] = None

    # ---------- status ----------
    def status(self) -> Dict:
        return {
            "warm": self.warm,
            "warmup_error": self.warmup_error,
            "llm": self.llm.name if self.llm else None,
            "qa_device": self.qa.device,
            "qa_runtime": self.qa.runtime,
            "ocr_backend": self.extractor.backend,
        }

    def warmup(self):
        """Load models up-front so the first upload isn't slow."""
        t0 = time.perf_counter()
        try:
            with self._lock:
                self.qa.warmup()
                if self.llm:
                    self.llm.warmup()
                else:
                    self.summarizer.warmup()
                    self.qgen.warmup()
            self.warm = True
            log.info(f"Warm-up finished in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # app still works; models load lazily later
            self.warmup_error = f"{type(e).__name__}: {e}"
            log.exception("Warm-up failed")

    # ---------- pipeline ----------
    def process(self, file_path: str, num_questions: int = 5, style: str = "mcq",
                ocr_backend: Optional[str] = None, progress: Progress = None) -> Dict:
        timings: Dict[str, float] = {}

        def stage(name, msg):
            if progress:
                progress(name, msg)

        with self._lock:
            if ocr_backend in ("easyocr", "handwriting"):
                self.extractor.backend = ocr_backend
            stage("extract", "Reading your notes")
            t = time.perf_counter()
            text = self.extractor.extract(file_path, progress=lambda m: stage("extract", m))
            timings["extract_ms"] = (time.perf_counter() - t) * 1000
            if len(text.strip()) < 30:
                raise ValueError(
                    "No readable text found. If these are handwritten notes, switch "
                    "'Notes type' to Handwritten and try again."
                )

            index = BM25Index(text)

            stage("summary", "Writing the summary")
            t = time.perf_counter()
            summary, summary_engine = self._summary(text)
            timings["summary_ms"] = (time.perf_counter() - t) * 1000

            stage("questions", "Creating practice questions")
            t = time.perf_counter()
            questions, q_engine = self._questions(text, num_questions, style)
            timings["questions_ms"] = (time.perf_counter() - t) * 1000

        return {
            "text": text,
            "index": index,
            "summary": summary,
            "questions": questions,
            "stats": dict(self.extractor.stats, chars=len(text)),
            "engines": {"summary": summary_engine, "questions": q_engine,
                        "ocr": self.extractor.backend},
            "timings": timings,
        }

    def _summary(self, text: str):
        if self.llm:
            try:
                return self.llm.summarize(text), f"{self.llm.name} (CPU, llama.cpp)"
            except Exception:
                log.exception("LLM summary failed; using T5 fallback")
        return self.summarizer.summarize(text), f"{self.summarizer.model_name} (CPU)"

    def _questions(self, text: str, n: int, style: str):
        if self.llm:
            try:
                qs = self.llm.questions(text, n, style)
                if qs:
                    return qs, f"{self.llm.name} (CPU, llama.cpp)"
            except Exception:
                log.exception("LLM questions failed; using T5-QG fallback")
        return self.qgen.generate(text, n), f"{self.qgen.model_name} (CPU)"

    def questions_only(self, text: str, n: int, style: str) -> Dict:
        with self._lock:
            qs, eng = self._questions(text, n, style)
        return {"questions": qs, "engine": eng}

    def ask(self, question: str, text: str, index: Optional[BM25Index]) -> Dict:
        """Grounded answering. The answer is always a span that the extractive
        QA model copied from the notes; the local LLM (if installed) only adds
        an explanation of that span. Nothing is answered when retrieval finds
        no matching passage or the QA model is not confident."""
        index = index or BM25Index(text)
        t = time.perf_counter()
        result = {
            "answer": "", "found": False, "score": 0.0, "evidence": "",
            "engine": "BM25 retrieval", "explanation": "", "explanation_engine": None,
            "reason": "",
        }
        passages: List[str] = index.top_passages(question, k=3, fallback=False)
        if not passages:
            result["reason"] = "no_matching_notes"
        else:
            with self._lock:
                span = self.qa.answer(question, text, passages=passages)
                result.update(score=span["score"], engine=span["engine"])
                if not span["found"]:
                    result["reason"] = "low_confidence"
                else:
                    result.update(answer=span["answer"], found=True, evidence=span["context"])
                    if self.llm:
                        try:
                            expl = self.llm.explain(question, span["answer"], span["context"], passages)
                            if expl:
                                result.update(explanation=expl,
                                              explanation_engine=f"{self.llm.name} (CPU, llama.cpp)")
                        except Exception:
                            log.exception("LLM explanation failed; showing the answer without it")
        result["latency_ms"] = (time.perf_counter() - t) * 1000
        return result
