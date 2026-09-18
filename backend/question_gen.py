"""
question_gen.py
Fallback practice-question generator used when no local LLM is installed.
Answer-aware QG with valhalla/t5-small-qg-hl: pick a short key phrase in a
sentence, wrap it in <hl> tokens, and ask the model for a question whose
answer is that phrase.

Sentences are sampled across the whole document (not just the start).
"""
import re
from typing import Dict, List, Optional

from config import SETTINGS
from utils import get_logger, split_sentences, timed

log = get_logger("question_gen")


class QuestionGenerator:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or SETTINGS.qg_model
        self._tok = None
        self._model = None
        self._nlp = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            log.info(f"Loading QG model: {self.model_name}")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).eval()
        return self._tok, self._model

    def _spacy(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:  # not installed / model missing
                log.warning("spaCy en_core_web_sm unavailable; using regex key-phrase extraction.")
                self._nlp = False
        return self._nlp

    def warmup(self):
        self._load()

    @staticmethod
    def _regex_candidates(sentence: str) -> List[str]:
        c = re.findall(r"\b[A-Z][a-zA-Z]{2,}(?:\s[A-Z][a-zA-Z]{2,}){0,3}\b", sentence)
        c += re.findall(r"\b\d+(?:\.\d+)?\s?(?:%|km|kg|m|s|years?|°C)?", sentence)
        return [x.strip() for x in c if x.strip()]

    def _candidates(self, sentence: str) -> List[str]:
        nlp = self._spacy()
        if not nlp:
            cands = self._regex_candidates(sentence)
        else:
            doc = nlp(sentence)
            cands = [e.text for e in doc.ents] + [
                ch.text for ch in doc.noun_chunks
                if ch.root.pos_ != "PRON" and ch.text.lower() not in {"it", "this", "they"}
            ]
        # prefer specific but short answers (1-5 words), skip the sentence's first word
        good = [c for c in dict.fromkeys(cands) if 1 <= len(c.split()) <= 5 and len(c) > 2]
        return sorted(good, key=lambda c: (-min(len(c.split()), 3), len(c)))

    def _generate_for_span(self, sentence: str, answer: str) -> str:
        tok, model = self._load()
        highlighted = sentence.replace(answer, f"<hl> {answer} <hl>", 1)
        inputs = tok(f"generate question: {highlighted}", return_tensors="pt",
                     truncation=True, max_length=256)
        import torch
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=48, num_beams=4)
        return tok.decode(ids[0], skip_special_tokens=True).strip()

    @timed
    def generate(self, text: str, num_questions: int = 5, style: str = "short") -> List[Dict]:
        sentences = [s for s in split_sentences(text) if 40 <= len(s) <= 400]
        if not sentences:
            return []
        # sample evenly across the document, with spares for failures
        want = min(len(sentences), num_questions * 3)
        step = len(sentences) / want
        pool = [sentences[int(i * step)] for i in range(want)]

        out: List[Dict] = []
        seen = set()
        for sentence in pool:
            if len(out) >= num_questions:
                break
            cands = self._candidates(sentence)
            if not cands:
                continue
            answer = cands[0]
            try:
                q = self._generate_for_span(sentence, answer)
            except Exception as e:  # keep going on a bad sentence
                log.warning(f"QG failed, skipping sentence: {e}")
                continue
            key = q.lower()
            if not q.endswith("?") or key in seen or answer.lower() in key:
                continue
            seen.add(key)
            out.append({"type": "short", "question": q, "answer": answer,
                        "explanation": sentence, "marks": 1})
        return out
