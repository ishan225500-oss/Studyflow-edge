"""
quantize_qa_qnn.py
Static QDQ quantisation for the Qualcomm QNN Execution Provider (Hexagon NPU).

Dynamic quantisation (quantize_dynamic) produces operators the NPU can't
run, so this uses ONNX Runtime's QNN-specific config: calibrated uint8
weights and uint16 activations (a common accuracy-preserving choice for
transformers on the Hexagon NPU).

    python scripts/quantize_qa_qnn.py
    -> models/qa/model.qdq.onnx   (the app picks it automatically)

Low-RAM laptops (8 GB): close other apps; if it still runs out of memory use
    python scripts/quantize_qa_qnn.py --stride 2 --max-samples 24

Then compare accuracy:  python scripts/evaluate_qa.py --compare
Needs: onnx, onnxruntime (>=1.18), transformers (tokenizer only).
"""
import argparse
import json

import numpy as np

import _common  # noqa: F401
import paths
from _common import ROOT
from retrieval import BM25Index


class QACalibrationReader:
    def __init__(self, tokenizer, pairs, seq_len, input_types):
        self.items = []
        for q, ctx in pairs:
            enc = tokenizer(q, ctx, truncation="only_second", max_length=seq_len,
                            padding="max_length", return_tensors="np")
            self.items.append({
                name: (enc["input_ids"] if "input_ids" in name else enc["attention_mask"]).astype(dt)
                for name, dt in input_types.items()
            })
        self._it = iter(self.items)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self.items)


def calibration_pairs(extra_file=None):
    samples = ROOT / "data" / "samples"
    notes = (samples / "sample_notes.txt").read_text(encoding="utf-8")
    if extra_file:
        notes += "\n" + open(extra_file, encoding="utf-8").read()
    idx = BM25Index(notes, chunk_chars=900)
    questions = [it["question"] for it in json.loads((samples / "qa_eval.json").read_text())["items"]]
    questions += ["What is " + c.split(".")[0][:60] + "?" for c in idx.chunks]
    pairs = []
    for q in questions:
        for p in idx.top_passages(q, k=2):
            pairs.append((q, p))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(paths.models_dir() / "qa" / "model.onnx"))
    ap.add_argument("--output", default=str(paths.models_dir() / "qa" / "model.qdq.onnx"))
    ap.add_argument("--calib-text", help="extra .txt notes to calibrate on (recommended)")
    ap.add_argument("--activations", choices=["uint8", "uint16"], default="uint16")
    ap.add_argument("--max-samples", type=int, default=40,
                    help="calibration pairs to use (fewer = less RAM and time)")
    ap.add_argument("--stride", type=int, default=4,
                    help="calibrate in groups of this many samples to cap RAM (8 GB laptops: 2-4)")
    args = ap.parse_args()

    import onnx
    from onnxruntime.quantization import QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
    from transformers import AutoTokenizer

    model_dir = paths.models_dir() / "qa"
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    m = onnx.load(args.input, load_external_data=False)
    input_types, seq_len = {}, None
    for inp in m.graph.input:
        elem = inp.type.tensor_type.elem_type
        input_types[inp.name] = np.int32 if elem == onnx.TensorProto.INT32 else np.int64
        seq_len = inp.type.tensor_type.shape.dim[1].dim_value
    print(f"Inputs {list(input_types)} seq_len={seq_len}")

    pre = model_dir / "model.preproc.onnx"
    changed = qnn_preprocess_model(args.input, str(pre), fuse_layernorm=True)
    src = str(pre) if changed else args.input

    pairs = calibration_pairs(args.calib_text)
    if len(pairs) > args.max_samples:
        step = len(pairs) / args.max_samples
        pairs = [pairs[int(i * step)] for i in range(args.max_samples)]
    print(f"Calibrating on {len(pairs)} question/passage pairs ...")
    reader = QACalibrationReader(tok, pairs, seq_len, input_types)
    act = QuantType.QUInt16 if args.activations == "uint16" else QuantType.QUInt8
    try:
        cfg = get_qnn_qdq_config(src, reader, activation_type=act, weight_type=QuantType.QUInt8,
                                 stride=args.stride)
    except TypeError:  # older onnxruntime without the stride option
        cfg = get_qnn_qdq_config(src, reader, activation_type=act, weight_type=QuantType.QUInt8)
    quantize(src, args.output, cfg)
    if changed:
        pre.unlink(missing_ok=True)

    import os
    print(f"Wrote {args.output} ({os.path.getsize(args.output) / 2**20:.0f} MB)")
    print("Next: python scripts/evaluate_qa.py --compare   and   python scripts/check_npu.py")


if __name__ == "__main__":
    main()
