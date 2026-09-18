"""
evaluate_qa.py
Question-answering quality on data/samples/qa_eval.json, using the same path
as the app: BM25 retrieval (no fallback passage) -> DistilBERT QA -> the
confidence threshold (STUDYFLOW_QA_MIN_SCORE).

Reports, separately:
- answerable questions: exact match, F1, how often an answer was given
- unanswerable questions: how often the app correctly said "not in your
  notes" (overall, and for near-miss vs off-topic questions)

    python scripts/evaluate_qa.py                 # whatever the app would use
    python scripts/evaluate_qa.py --compare       # PyTorch vs ONNX FP32 vs ONNX QDQ (CPU)
    python scripts/evaluate_qa.py --onnx models/qa/model.qdq.onnx   # on NPU if available
    python scripts/evaluate_qa.py --onnx models/qa/model.qdq.onnx --sweep
          # try thresholds 0.05..0.90 and suggest STUDYFLOW_QA_MIN_SCORE
"""
import argparse
import json
import time

import _common
from _common import ROOT, machine_info, save_result
import paths
from config import SETTINGS
from qa_metrics import sweep as run_sweep, summarise
from qa_module import QAModule
from retrieval import BM25Index


def load_set(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    text = (path.parent / data["context_file"]).read_text(encoding="utf-8")
    return data["items"], text


def run_model(qa: QAModule, items, text):
    index = BM25Index(text)
    qa.answer("warm up?", "warm up text.")
    rows, lat = [], []
    for it in items:
        passages = index.top_passages(it["question"], k=3, fallback=False)
        row = {"question": it["question"], "kind": it.get("kind") or
               ("answerable" if it["answers"] else "unanswerable"),
               "answers": it["answers"], "retrieved": bool(passages),
               "raw_pred": "", "raw_score": 0.0}
        if passages:
            t = time.perf_counter()
            r = qa.answer(it["question"], text, passages=passages)
            lat.append((time.perf_counter() - t) * 1000)
            row.update(raw_pred=r["answer"], raw_score=round(r["score"], 4))
        rows.append(row)
    lat.sort()
    latency = {
        "latency_ms_p50": round(lat[len(lat) // 2], 1) if lat else None,
        "latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
    }
    return rows, latency


def evaluate(qa: QAModule, label: str, items, text, threshold: float, do_sweep: bool) -> dict:
    rows, latency = run_model(qa, items, text)
    s = summarise(rows, threshold)
    res = {"label": label, "device": qa.device, **s, **latency, "items": rows}
    a, u = s["answerable"], s["unanswerable"]
    print(f"{label:<24} {qa.device:<30} thr {threshold:.2f}  "
          f"EM {s['exact_match']:5.1f}  F1 {s['f1']:5.1f} | answerable F1 {a['f1']} "
          f"(answered {a['answered_rate']}%) | not-in-notes rejected {u['rejection_rate']}% "
          f"of {u['n']} | p50 {latency['latency_ms_p50']} ms")
    if do_sweep:
        sw = run_sweep(rows)
        res["sweep"] = sw
        print(f"\n  {'thr':>5} {'correct%':>9} {'answered%':>10} {'rejected%':>10} "
              f"{'near-miss%':>11} {'balanced':>9}")
        for t in sw["table"]:
            print(f"  {t['threshold']:>5.2f} {t['answerable']['correct_rate']!s:>9} "
                  f"{t['answerable']['answered_rate']!s:>10} "
                  f"{t['unanswerable']['rejection_rate']!s:>10} "
                  f"{t['unanswerable'].get('near_miss_rejection_rate')!s:>11} {t['balanced']!s:>9}")
        b = sw["best"]
        print(f"\n  Suggested: STUDYFLOW_QA_MIN_SCORE={b['threshold']:.2f} "
              f"(correct {b['answerable']['correct_rate']}%, rejected "
              f"{b['unanswerable']['rejection_rate']}%). Choose it on the runtime the app "
              "ships (QDQ scores can differ from FP32), and check it on questions from "
              "your own notes: this set is small.\n")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="PyTorch vs ONNX FP32 vs ONNX QDQ, all on CPU")
    ap.add_argument("--onnx", help="evaluate this ONNX file (NPU if available)")
    ap.add_argument("--cpu", action="store_true", help="never use the NPU")
    ap.add_argument("--sweep", action="store_true", help="try thresholds and suggest one")
    ap.add_argument("--threshold", type=float, default=SETTINGS.qa_min_score,
                    help=f"confidence threshold to report at (default {SETTINGS.qa_min_score})")
    ap.add_argument("--set", default=str(ROOT / "data" / "samples" / "qa_eval.json"),
                    help="labelled QA set (same format as data/samples/qa_eval.json)")
    args = ap.parse_args()

    from pathlib import Path
    items, text = load_set(Path(args.set))
    kw = dict(items=items, text=text, threshold=args.threshold, do_sweep=args.sweep)

    results = []
    if args.compare:
        results.append(evaluate(QAModule(runtime="torch"), "PyTorch FP32", **kw))
        qa_dir = paths.models_dir() / "qa"
        for name, label in (("model.onnx", "ONNX FP32"), ("model.qdq.onnx", "ONNX QDQ (NPU format)")):
            if (qa_dir / name).exists():
                results.append(evaluate(QAModule(runtime="onnx", onnx_path=qa_dir / name,
                                                 prefer_npu=False), label + " / CPU", **kw))
    elif args.onnx:
        results.append(evaluate(QAModule(runtime="onnx", onnx_path=args.onnx,
                                         prefer_npu=not args.cpu), "ONNX", **kw))
    else:
        results.append(evaluate(QAModule(), "App default", **kw))
    save_result("qa-eval", {"machine": machine_info(), "eval_set": args.set, "results": results})


if __name__ == "__main__":
    main()
