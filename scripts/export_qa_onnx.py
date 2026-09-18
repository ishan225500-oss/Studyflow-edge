"""
export_qa_onnx.py
Export the DistilBERT QA model to a static-shape ONNX graph the Snapdragon
NPU can run (QNN needs fixed shapes), then check it matches PyTorch.

    python scripts/export_qa_onnx.py            # -> models/qa/model.onnx (+ tokenizer)
    python scripts/export_qa_onnx.py --seq-len 256

Inputs are int32 (input_ids, attention_mask) of shape [1, seq_len];
outputs are start_logits and end_logits of shape [1, seq_len].
Needs: torch, transformers, onnx, onnxruntime.
"""
import argparse
import inspect

import numpy as np

import _common  # noqa: F401
import paths
from config import SETTINGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=SETTINGS.qa_model)
    ap.add_argument("--seq-len", type=int, default=SETTINGS.qa_seq_len)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    out_dir = paths.models_dir() / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    model.eval()

    class StaticQA(torch.nn.Module):
        """int32 inputs (NPU friendly) cast to int64 inside the graph."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids.long(), attention_mask=attention_mask.long())
            return out.start_logits, out.end_logits

    wrapper = StaticQA(model).eval()
    enc = tok("What produces ATP?", "The mitochondria produce ATP through respiration.",
              max_length=args.seq_len, padding="max_length", truncation=True, return_tensors="pt")
    ids = enc["input_ids"].to(torch.int32)
    mask = enc["attention_mask"].to(torch.int32)

    kwargs = dict(input_names=["input_ids", "attention_mask"],
                  output_names=["start_logits", "end_logits"],
                  opset_version=args.opset, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False   # classic TorchScript exporter: stable static graph
    print(f"Exporting {args.model} -> {out_path} (seq_len={args.seq_len}, opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(wrapper, (ids, mask), str(out_path), **kwargs)
        ref_s, ref_e = wrapper(ids, mask)

    tok.save_pretrained(out_dir)

    import onnx
    onnx.checker.check_model(str(out_path))
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    s, e = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
    diff = max(np.abs(s - ref_s.numpy()).max(), np.abs(e - ref_e.numpy()).max())
    print(f"Max |ONNX - PyTorch| logit difference: {diff:.2e}")
    if diff > 1e-3:
        raise SystemExit("Export mismatch is too large; do not ship this model.")
    size_mb = out_path.stat().st_size / 2**20
    print(f"OK: {out_path} ({size_mb:.0f} MB). Next: scripts/quantize_qa_qnn.py")


if __name__ == "__main__":
    main()
