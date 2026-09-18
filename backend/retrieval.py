"""
retrieval.py
Tiny dependency-free BM25 index over chunks of the user's notes.
Used to pick the most relevant passages before question answering, so
long documents work with small context windows (384-token QA model,
few-thousand-token local LLM).
"""
import math
import re
from collections import Counter
from typing import List, Tuple

from utils import chunk_text

_TOKEN = re.compile(r"[A-Za-z0-9\u0900-\u0D7F]+")
STOPWORDS = set(
    "a an the is are was were be been being of in on at to for from by with and or not "
    "what which who whom whose when where why how does do did can could should would "
    "this that these those it its as into about explain describe define give list "
    "me my i you your please tell".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in (w.lower() for w in _TOKEN.findall(text)) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, text: str, chunk_chars: int = 900, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunk_text(text, max_chars=chunk_chars) if text.strip() else []
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        q = tokenize(query)
        scores = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in q:
                tf = d.get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                s += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
            scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    def top_passages(self, query: str, k: int = 3, max_chars: int = 2400,
                     fallback: bool = True) -> List[str]:
        """Best-matching chunks, most relevant first.

        fallback=True returns the first chunk when no chunk shares a word with
        the query (useful for calibration text). Question answering passes
        fallback=False: no shared words means the notes don't cover the
        question, and the models must not be handed an unrelated passage."""
        hits = [i for i, s in self.search(query, k) if s > 0]
        if not hits and self.chunks and fallback:
            hits = [0]
        out, total = [], 0
        for i in hits:
            c = self.chunks[i]
            if total + len(c) > max_chars and out:
                break
            out.append(c)
            total += len(c)
        return out
