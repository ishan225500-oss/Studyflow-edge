"""
llm.py
Optional local LLM (llama.cpp, GGUF weights) for higher-quality summaries,
exam-style questions, and explanations of answers that the extractive QA
model has already found in the notes (the LLM never supplies the answer). Runs fully offline on the
Snapdragon Oryon CPU via llama-cpp-python's native ARM64 build.

NPU path: the same prompts are designed to be served by Qualcomm AI Hub's
Snapdragon-optimised LLMs (e.g. Llama 3.2 3B) through Qualcomm Genie; see
README "Roadmap". This module is the runtime we measured ourselves.
"""
import json
import os
import re
import threading
from typing import Dict, List, Optional

from config import SETTINGS
from utils import chunk_text, get_logger, timed

log = get_logger("llm")

SYSTEM = (
    "You are StudyFlow, a careful study assistant for university and competitive-exam "
    "students in India. Use ONLY the student's notes provided. Be concise and accurate. "
    "Never invent facts that are not in the notes."
)

NOT_FOUND = "NOT_IN_NOTES"

STYLE_INSTRUCTIONS = {
    "mcq": (
        'Write {n} multiple-choice questions. Each item: {{"type":"mcq","question":str,'
        '"options":[4 short strings],"answer":str (exactly one of the options),'
        '"explanation":str (one sentence),"marks":1}}'
    ),
    "short": (
        'Write {n} short-answer questions. Each item: {{"type":"short","question":str,'
        '"answer":str (1-2 sentences),"explanation":"","marks":2}}'
    ),
    "exam": (
        'Write {n} university-exam questions: mix 2-mark and 5-mark questions. Each item: '
        '{{"type":"long","question":str,"answer":str (model answer as key points separated '
        'by newlines; 2-mark = 2 points, 5-mark = 4-5 points),"explanation":"","marks":2 or 5}}'
    ),
}


def available() -> bool:
    if not SETTINGS.use_llm or SETTINGS.llm_gguf_path() is None:
        return False
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


def parse_json_block(text: str):
    """Extract the first JSON object/array from model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("model did not return valid JSON")


def normalise_questions(data, n: int) -> List[Dict]:
    items = data.get("questions", []) if isinstance(data, dict) else data
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        q = str(it.get("question", "")).strip()
        a = str(it.get("answer", "")).strip()
        if not q or not a:
            continue
        qtype = it.get("type", "short")
        opts = [str(o).strip() for o in it.get("options", []) if str(o).strip()] if qtype == "mcq" else []
        if qtype == "mcq":
            if len(opts) < 2:
                continue
            if a not in opts:  # model sometimes answers "B" or a paraphrase
                letter = a.strip().upper().rstrip(").")
                if len(letter) == 1 and "A" <= letter <= "D" and ord(letter) - 65 < len(opts):
                    a = opts[ord(letter) - 65]
                else:
                    continue
        try:
            marks = int(it.get("marks", 1))
        except (TypeError, ValueError):
            marks = 1
        out.append({
            "type": qtype, "question": q, "answer": a, "options": opts,
            "explanation": str(it.get("explanation", "")).strip(), "marks": marks,
        })
    return out[:n]


class LocalLLM:
    def __init__(self):
        self.path = SETTINGS.llm_gguf_path()
        self._llm = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.path.stem if self.path else "none"

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama
            threads = SETTINGS.llm_threads or max(1, (os.cpu_count() or 4) - 1)
            log.info(f"Loading LLM {self.path.name} (ctx={SETTINGS.llm_ctx}, threads={threads})")
            self._llm = Llama(
                model_path=str(self.path), n_ctx=SETTINGS.llm_ctx,
                n_threads=threads, verbose=False,
            )
        return self._llm

    def warmup(self):
        self.chat("Reply with OK.", max_tokens=4)

    def chat(self, user: str, max_tokens: int = 512, temperature: float = 0.2,
             json_mode: bool = False) -> str:
        llm = self._load()
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        with self._lock:
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=temperature, **kwargs,
            )
        return out["choices"][0]["message"]["content"].strip()

    # budget in characters for notes inside one prompt (~3.5 chars/token)
    def _budget(self, reserve_tokens: int) -> int:
        return max(1500, int((SETTINGS.llm_ctx - reserve_tokens) * 3.2))

    @timed
    def summarize(self, text: str) -> str:
        budget = self._budget(900)
        if len(text) > budget:
            chunks = chunk_text(text, max_chars=budget)
            step = max(1, len(chunks) // 5)
            notes = []
            for c in chunks[::step][:5]:
                notes.append(self.chat(
                    f"Notes section:\n{c}\n\nList the 3-5 most exam-relevant facts as short bullets.",
                    max_tokens=220))
            text = "\n".join(notes)
        return self.chat(
            f"Student notes:\n{text}\n\nWrite a revision summary: a one-line topic statement, "
            "then 5-8 bullet points ('- ') covering key definitions, formulas and facts.",
            max_tokens=450,
        )

    @timed
    def questions(self, text: str, n: int = 5, style: str = "mcq") -> List[Dict]:
        style = style if style in STYLE_INSTRUCTIONS else "mcq"
        budget = self._budget(1400)
        if len(text) > budget:  # sample sections spread over the document
            chunks = chunk_text(text, max_chars=budget // 3)
            step = max(1, len(chunks) // 3)
            text = "\n...\n".join(chunks[::step][:3])
        prompt = (
            f"Student notes:\n{text}\n\n{STYLE_INSTRUCTIONS[style].format(n=n)}\n"
            "Cover different parts of the notes. Return JSON: {\"questions\": [ ... ]}"
        )
        for attempt in range(2):
            raw = self.chat(prompt, max_tokens=1400, json_mode=True, temperature=0.3 + 0.2 * attempt)
            try:
                qs = normalise_questions(parse_json_block(raw), n)
                if qs:
                    return qs
            except ValueError as e:
                log.warning(f"Question JSON parse failed (attempt {attempt + 1}): {e}")
        return []

    @timed
    def explain(self, question: str, answer: str, evidence: str, passages: List[str]) -> str:
        """Explain an answer that the extractive QA model already found in the
        notes. The LLM never supplies the answer itself; it only explains the
        verified span, and it may decline if the notes don't support it.
        Returns "" when there is nothing safe to show."""
        notes = "\n---\n".join(passages)
        reply = self.chat(
            f"Relevant parts of the student's notes:\n{notes}\n\n"
            f"Question: {question}\n"
            f"Answer found in the notes: {answer}\n"
            f"Supporting line: {evidence}\n\n"
            "In 2-4 sentences, explain this answer to the student using only the notes above. "
            "Do not change the answer and do not add facts that are not in the notes. "
            f"If the notes do not support this answer, reply with exactly {NOT_FOUND}.",
            max_tokens=250,
        )
        return clean_explanation(reply)


def clean_explanation(reply: str) -> str:
    """Drop empty replies and any reply that declines (contains NOT_IN_NOTES)."""
    reply = (reply or "").strip()
    if not reply or NOT_FOUND in reply:
        return ""
    return reply
