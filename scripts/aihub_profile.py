"""
aihub_profile.py
Compile the QA model with Qualcomm AI Hub and profile it on a real
Snapdragon device in Qualcomm's device cloud, so the reported NPU numbers
are measured, not estimated. No Snapdragon laptop needed.

Setup (once):
    pip install qai-hub
    qai-hub configure --api_token <YOUR_TOKEN>     # token from https://aihub.qualcomm.com

Usage:
    python scripts/aihub_profile.py --list-devices
    python scripts/aihub_profile.py                              # QDQ model if present
    python scripts/aihub_profile.py --model models/qa/model.onnx --device "Snapdragon X Elite CRD"
    python scripts/aihub_profile.py --verify                     # also check outputs on device

Writes results/aihub-profile-<time>.json and downloads the compiled model
to models/qa/aihub/.
"""
import argparse

import numpy as np

import _common  # noqa: F401
from _common import save_result
import paths
from config import SETTINGS

DEFAULT_DEVICE = "Snapdragon X Elite CRD"


def pick(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def summarise_profile(profile: dict) -> dict:
    summary = profile.get("execution_summary", {}) if isinstance(profile, dict) else {}
    layers = profile.get("execution_detail", []) if isinstance(profile, dict) else []
    units = {}
    for layer in layers:
        u = layer.get("compute_unit", "unknown")
        units[u] = units.get(u, 0) + 1
    inf_us = summary.get("estimated_inference_time")
    mem_b = summary.get("estimated_inference_peak_memory")
    return {
        "inference_ms": round(inf_us / 1000, 2) if isinstance(inf_us, (int, float)) else None,
        "peak_memory_mb": round(mem_b / 2**20, 1) if isinstance(mem_b, (int, float)) else None,
        "first_load_ms": (lambda v: round(v / 1000, 1) if isinstance(v, (int, float)) else None)(
            summary.get("first_load_time")),
        "warm_load_ms": (lambda v: round(v / 1000, 1) if isinstance(v, (int, float)) else None)(
            summary.get("warm_load_time")),
        "layers_by_compute_unit": units,
    }


def sample_inputs(seq_len: int):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(paths.models_dir() / "qa"), use_fast=True)
    enc = tok("Which organelle produces ATP?",
              "The mitochondria are known as the powerhouse of the cell because they produce ATP.",
              max_length=seq_len, padding="max_length", truncation=True, return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


def main():
    ap = argparse.ArgumentParser()
    qa_dir = paths.models_dir() / "qa"
    default_model = qa_dir / ("model.qdq.onnx" if (qa_dir / "model.qdq.onnx").exists() else "model.onnx")
    ap.add_argument("--model", default=str(default_model))
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--runtime", default="onnx",
                    help="AI Hub --target_runtime, e.g. onnx, precompiled_qnn_onnx, qnn_context_binary")
    ap.add_argument("--seq-len", type=int, default=SETTINGS.qa_seq_len)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--verify", action="store_true", help="run inference on device and compare to local CPU")
    args = ap.parse_args()

    import qai_hub as hub

    if args.list_devices:
        for d in hub.get_devices():
            print(d)
        return

    device = hub.Device(args.device)
    L = args.seq_len
    specs = {"input_ids": ((1, L), "int32"), "attention_mask": ((1, L), "int32")}
    print(f"Compiling {args.model} for '{args.device}' (target_runtime={args.runtime}) ...")
    compile_job = hub.submit_compile_job(
        model=args.model, device=device, input_specs=specs,
        options=f"--target_runtime {args.runtime}", name="studyflow-qa-compile",
    )
    target = compile_job.get_target_model()
    if target is None:
        raise SystemExit(f"Compile failed. See the job page: {compile_job.url}")

    out_dir = qa_dir / "aihub"
    out_dir.mkdir(exist_ok=True)
    downloaded = target.download(str(out_dir / f"model.{args.runtime}"))
    print(f"Compiled model saved to {downloaded}")

    print("Profiling on device ...")
    profile_job = hub.submit_profile_job(model=target, device=device, name="studyflow-qa-profile")
    profile = profile_job.download_profile()
    summary = summarise_profile(profile)
    print(f"  inference: {summary['inference_ms']} ms   peak memory: {summary['peak_memory_mb']} MB")
    print(f"  layers by compute unit: {summary['layers_by_compute_unit']}")

    result = {
        "model": args.model, "device": args.device, "target_runtime": args.runtime, "seq_len": L,
        "compile_job": compile_job.url, "profile_job": profile_job.url,
        "summary": summary, "raw_profile": profile,
    }

    if args.verify:
        import onnxruntime as ort
        ids, mask = sample_inputs(L)
        inf_job = hub.submit_inference_job(
            model=target, device=device,
            inputs={"input_ids": [ids], "attention_mask": [mask]}, name="studyflow-qa-verify",
        )
        dev_out = inf_job.download_output_data()
        cpu = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        cpu_s, cpu_e = cpu.run(None, {"input_ids": ids, "attention_mask": mask})
        dev_s = np.asarray(dev_out["start_logits"][0]).reshape(cpu_s.shape)
        dev_e = np.asarray(dev_out["end_logits"][0]).reshape(cpu_e.shape)
        same = int(dev_s.argmax()) == int(cpu_s.argmax()) and int(dev_e.argmax()) == int(cpu_e.argmax())
        result["verify"] = {
            "inference_job": inf_job.url,
            "same_answer_span_as_cpu": same,
            "max_abs_logit_diff": float(max(np.abs(dev_s - cpu_s).max(), np.abs(dev_e - cpu_e).max())),
        }
        print(f"  on-device answer span matches CPU: {same}")

    save_result("aihub-profile", result)


if __name__ == "__main__":
    main()
