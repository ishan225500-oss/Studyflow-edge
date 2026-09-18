"""
check_npu.py
Run ON the Snapdragon HP PC. Proves the QA model is fully offloaded to the
Hexagon NPU (CPU fallback disabled) and measures NPU vs CPU latency.

    python scripts/check_npu.py
    python scripts/check_npu.py --model models/qa/model.qdq.onnx --runs 50
"""
import argparse
import statistics
import time

import numpy as np

import _common  # noqa: F401
from _common import machine_info, save_result
import paths
from utils import build_onnxruntime_session


def bench(session, feeds, runs):
    for _ in range(3):
        session.run(None, feeds)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"mean_ms": round(statistics.mean(times), 2), "p50_ms": round(times[len(times) // 2], 2),
            "p90_ms": round(times[int(len(times) * 0.9) - 1], 2), "runs": runs}


def main():
    qa_dir = paths.models_dir() / "qa"
    default = qa_dir / ("model.qdq.onnx" if (qa_dir / "model.qdq.onnx").exists() else "model.onnx")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(default))
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    info = machine_info()
    print("Machine:", info)
    if info.get("python_platform") == "win-amd64" and "ARM" in (info.get("processor") or "").upper():
        print("WARNING: x64 Python running under emulation on an ARM64 PC; CPU numbers will be slow.")

    cpu, _ = build_onnxruntime_session(args.model, prefer_npu=False)
    feeds = {}
    for inp in cpu.get_inputs():
        L = inp.shape[1]
        dtype = np.int32 if "int32" in inp.type else np.int64
        feeds[inp.name] = (np.random.randint(1000, 5000, size=(1, L)) if "input_ids" in inp.name
                           else np.ones((1, L))).astype(dtype)

    result = {"machine": info, "model": args.model, "cpu": bench(cpu, feeds, args.runs)}
    print(f"CPU: {result['cpu']}")

    try:
        t = time.perf_counter()
        npu, provider = build_onnxruntime_session(args.model, prefer_npu=True, npu_only=True)
        result["npu_session_create_ms"] = round((time.perf_counter() - t) * 1000, 1)
        result["npu"] = bench(npu, feeds, args.runs)
        result["fully_offloaded_to_npu"] = True
        result["speedup_vs_cpu"] = round(result["cpu"]["p50_ms"] / result["npu"]["p50_ms"], 2)
        print(f"NPU ({provider}), CPU fallback disabled: {result['npu']}")
        print(f"Speed-up vs CPU (p50): {result['speedup_vs_cpu']}x")
    except Exception as e:
        result["fully_offloaded_to_npu"] = False
        result["npu_error"] = f"{type(e).__name__}: {e}"
        print(f"NPU run failed: {result['npu_error']}")
        print("Tip: use the QDQ model from scripts/quantize_qa_qnn.py and onnxruntime-qnn.")

    save_result("npu-check", result)


if __name__ == "__main__":
    main()
