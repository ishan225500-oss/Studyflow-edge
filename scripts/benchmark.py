"""
benchmark.py
End-to-end latency + peak memory for the real app pipeline, with model
loading measured separately from inference (warm-up excluded).

    python scripts/benchmark.py --doc data/samples/sample_notes.txt
    python scripts/benchmark.py --doc lecture.pdf --runs 3 --style exam

Writes results/benchmark-<time>.json and prints a Markdown table you can
paste into the README / slides.
"""
import argparse
import statistics
import threading
import time

import _common  # noqa: F401
from _common import machine_info, save_result


class PeakRSS:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self):
        try:
            import psutil
            proc = psutil.Process()

            def loop():
                while not self._stop.is_set():
                    self.peak = max(self.peak, proc.memory_info().rss)
                    time.sleep(0.05)
            threading.Thread(target=loop, daemon=True).start()
        except ImportError:
            self.peak = None
        return self

    def __exit__(self, *exc):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--question", default="Which organelle produces ATP?")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--style", default="mcq", choices=["mcq", "short", "exam"])
    ap.add_argument("--ocr", default=None, choices=["easyocr", "handwriting"])
    args = ap.parse_args()

    from engine import StudyEngine

    with PeakRSS() as mem:
        engine = StudyEngine()
        t = time.perf_counter()
        engine.warmup()
        load_s = time.perf_counter() - t
        if engine.warmup_error:
            raise SystemExit(f"Model loading failed: {engine.warmup_error}")

        runs = []
        for i in range(args.runs):
            res = engine.process(args.doc, num_questions=5, style=args.style, ocr_backend=args.ocr)
            t = time.perf_counter()
            ans = engine.ask(args.question, res["text"], res["index"])
            res["timings"]["ask_ms"] = (time.perf_counter() - t) * 1000
            runs.append(res["timings"])
            print(f"run {i + 1}: " + ", ".join(f"{k}={v:.0f}" for k, v in res["timings"].items()))

    if not ans["found"]:
        print(f"\nWARNING: the notes don't answer {args.question!r} ({ans.get('reason')}), so "
              "'Answer a question' only timed retrieval. Pass --question with one they do answer.")
    ask_engine = ans["engine"] + (f" + explanation by {ans['explanation_engine']}"
                                  if ans.get("explanation_engine") else "")

    stages = ["extract_ms", "summary_ms", "questions_ms", "ask_ms"]
    med = {s: round(statistics.median(r[s] for r in runs), 1) for s in stages}
    engines = res["engines"]
    table = [
        "| Stage | Median latency | Runs on |",
        "|---|---|---|",
        f"| Model loading (one-time) | {load_s:.1f} s | |",
        f"| Read notes ({res['stats']['pages']} pages, {res['stats']['ocr_pages']} OCR) | {med['extract_ms']:.0f} ms | {engines['ocr']} |",
        f"| Summary | {med['summary_ms']:.0f} ms | {engines['summary']} |",
        f"| 5 practice questions ({args.style}) | {med['questions_ms']:.0f} ms | {engines['questions']} |",
        f"| Answer a question | {med['ask_ms']:.0f} ms | {ask_engine} |",
    ]
    if mem.peak:
        table.append(f"| Peak memory (whole app) | {mem.peak / 2**20:.0f} MB | |")
    print("\n" + "\n".join(table))

    save_result("benchmark", {
        "machine": machine_info(), "doc": args.doc, "style": args.style,
        "model_load_s": round(load_s, 2), "median_ms": med, "runs": runs,
        "engines": engines, "qa_engine": ans["engine"], "stats": res["stats"],
        "peak_rss_mb": round(mem.peak / 2**20) if mem.peak else None,
        "sample_answer": ans.get("answer"), "answer_found": ans["found"],
        "sample_explanation": ans.get("explanation"), "ask_engine": ask_engine, "markdown": "\n".join(table),
    })


if __name__ == "__main__":
    main()
