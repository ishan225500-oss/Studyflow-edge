"""
qa_metrics.py
Scoring for scripts/evaluate_qa.py, kept free of model code so it can be
unit-tested.

Each evaluated row records what the QA model returned *before* the
confidence threshold is applied, so one model run can be re-scored at any
threshold:

    {"question", "kind", "answers", "retrieved", "raw_pred", "raw_score"}

kind is "answerable", "near_miss" (not in the notes, but shares keywords, so
only the threshold can reject it) or "off_topic" (no shared keywords).
"""
import re
import string
from collections import Counter
from typing import Dict, Iterable, List

CORRECT_F1 = 0.5   # an answered question counts as correct at F1 >= 0.5
DEFAULT_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]   # 0.05 .. 0.90


def norm(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def answered(row: Dict, threshold: float) -> bool:
    return bool(row["retrieved"] and row["raw_pred"] and row["raw_score"] >= threshold)


def _rate(num: float, den: int):
    return round(100 * num / den, 1) if den else None


def summarise(rows: List[Dict], threshold: float) -> Dict:
    em_sum = f1_sum = 0.0
    ans_n = ans_em = ans_f1 = ans_answered = ans_correct = 0.0
    rej = Counter()
    tot = Counter()
    for r in rows:
        said = answered(r, threshold)
        pred = r["raw_pred"] if said else ""
        if r["answers"]:
            em = float(said and max(norm(pred) == norm(g) for g in r["answers"]))
            f = max(f1(pred, g) for g in r["answers"]) if said else 0.0
            ans_n += 1
            ans_em += em
            ans_f1 += f
            ans_answered += said
            ans_correct += said and f >= CORRECT_F1
        else:
            em = f = float(not said)          # SQuAD 2.0 convention
            kind = r.get("kind") or "unanswerable"
            tot[kind] += 1
            rej[kind] += not said
        em_sum += em
        f1_sum += f
    n = len(rows)
    un_n = sum(tot.values())
    un_rej = sum(rej.values())
    ans_n_int = int(ans_n)
    correct_rate = _rate(ans_correct, ans_n_int)
    rejection_rate = _rate(un_rej, un_n)
    parts = [x for x in (correct_rate, rejection_rate) if x is not None]
    return {
        "threshold": threshold,
        "n": n,
        "exact_match": _rate(em_sum, n),
        "f1": _rate(f1_sum, n),
        "answerable": {
            "n": ans_n_int,
            "exact_match": _rate(ans_em, ans_n_int),
            "f1": _rate(ans_f1, ans_n_int),
            "answered_rate": _rate(ans_answered, ans_n_int),
            "correct_rate": correct_rate,
        },
        "unanswerable": {
            "n": un_n,
            "rejection_rate": rejection_rate,
            "false_answer_rate": _rate(un_n - un_rej, un_n),
            **{f"{k}_rejection_rate": _rate(rej[k], tot[k]) for k in sorted(tot)},
        },
        # mean of "answered correctly" and "refused correctly": what the sweep optimises
        "balanced": round(sum(parts) / len(parts), 1) if parts else None,
    }


def sweep(rows: List[Dict], thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> Dict:
    table = [summarise(rows, t) for t in thresholds]
    # Highest balanced score; on a tie prefer refusing more (safer for students).
    best = max(table, key=lambda s: (s["balanced"] or 0,
                                     s["unanswerable"]["rejection_rate"] or 0))
    return {"table": table, "best": best}
