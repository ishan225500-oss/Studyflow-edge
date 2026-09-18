# Start here

A short guide for anyone evaluating StudyFlow Edge. Everything below can be run and checked; nothing here asks you to take a claim on trust.

**StudyFlow Edge** is a study assistant that runs entirely on a Snapdragon laptop. Its distinguishing feature is that answers are *copied* from the student's own notes rather than generated: the local LLM may only explain an answer that an extractive model already found, and the app says so when the notes don't cover a question.

---

## 1. See it work (5 minutes)

**Fastest:** download `StudyFlowEdge-windows-*.zip` from [Releases](../../releases), unzip, run `StudyFlowEdge.exe`, click **Try sample notes**. Models are bundled, so no internet or Python is needed. Disconnect the Wi-Fi first if you'd like to check the offline claim.

**From source:**

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -t .     # 50 tests, no model weights needed, ~1 second
python scripts/download_models.py             # once, needs internet
python app/desktop.py
```

---

## 2. The thing worth testing first

Load the sample notes and ask these three questions in order:

| Question | Expected behaviour | Why it matters |
|---|---|---|
| *Which organelle produces ATP?* | A short answer copied from the notes, the source line quoted beneath it, and a separately labelled explanation | The answer is extracted, not generated |
| *What is the capital of Australia?* | "Your notes don't seem to cover this", returned almost instantly | No passage shares a word with the question, so neither model runs |
| *Who discovered mitochondria?* | The same refusal | The hard case: the notes mention mitochondria but never answer this, so only the confidence threshold can reject it |

The third question is the one most retrieval-augmented demos get wrong.

---

## 3. Claims, and how to check each one

| Claim | Command | What it proves |
|---|---|---|
| The QA model runs fully on the Hexagon NPU | `python scripts/check_npu.py` | Creates the session with CPU fallback disabled, so a successful run means every operation executed on the NPU. Also reports NPU vs CPU latency |
| ...measured on real silicon | `python scripts/aihub_profile.py --verify` | Compiles and profiles on a physical Snapdragon X Elite through Qualcomm AI Hub, and reports the compute unit used by each layer |
| Quantisation didn't wreck accuracy | `python scripts/evaluate_qa.py --compare` | PyTorch vs ONNX FP32 vs quantised QDQ, on the same labelled set |
| The app refuses questions its notes don't answer | `python scripts/evaluate_qa.py --onnx models/qa/model.qdq.onnx` | Reports accuracy on answerable questions and the rejection rate on unanswerable ones, split into near-miss and off-topic |
| The threshold was chosen, not guessed | `python scripts/evaluate_qa.py --sweep` | Sweeps thresholds from 0.05 to 0.90 and shows the trade-off behind the shipped value |
| The ONNX export matches PyTorch | `python scripts/export_qa_onnx.py` | Fails the export if any logit differs by more than 1e-3 |
| End-to-end speed and memory | `python scripts/benchmark.py --doc <pdf> --question "<q>"` | Median latency per stage and peak memory for the whole app |

Every script writes its raw output to `results/`, including the machine it ran on.

---

## 4. What runs where

| Task | Model | Runtime |
|---|---|---|
| Question answering | DistilBERT SQuAD, static-shape ONNX, QDQ quantised | **Hexagon NPU** — ONNX Runtime + QNN EP |
| Summary, practice questions, explanations | Llama 3.2 3B Instruct, GGUF Q4_K_M | Oryon CPU — llama.cpp |
| Printed and handwriting OCR | EasyOCR, TrOCR | CPU |

Only the QA model is on the NPU today. The interface labels which model and device produced each answer.

---

## 5. How the grounding is enforced

1. **Retrieval gate** — BM25 picks the three best passages; if none shares a word with the question, the app answers "not in your notes" and no model runs.
2. **Extraction** — the QA model copies a span from those passages. That span is always the answer, shown with its source line.
3. **Confidence gate** — below the calibrated threshold (`STUDYFLOW_QA_MIN_SCORE`), the app reports that the notes don't cover the question.
4. **Explanation only** — the LLM receives the verified span and may only explain it. It cannot replace the answer, and isn't called when there is no answer.

Tests covering this: `tests/test_grounding.py` — the LLM can't override a verified answer, off-topic questions never reach the models, and an LLM failure still leaves the answer intact.

---

## 6. Privacy, checkable

- The server binds to `127.0.0.1` and refuses requests addressed to any other host name, which blocks DNS rebinding. State-changing requests from other websites are refused. See `tests/test_security.py`.
- Uploaded files are deleted as soon as they're read; "Clear notes" wipes the session from memory.
- When model weights are present, the Hugging Face libraries are forced into offline mode.
- No AI service is called at any point. Switch off the Wi-Fi and the app behaves identically.

---

## 7. Known limits

- The QA model is trained on SQuAD 1.1, so "not in your notes" rests on a calibrated threshold rather than a native no-answer output. A SQuAD 2.0 model is the upgrade path.
- Retrieval is lexical, so a question worded very differently from the notes may not find the right passage.
- Only the QA model runs on the NPU; OCR and generation use the CPU.
- English is the tested language, though the tokenizer handles Indic scripts.

---

## 8. Where things are

```
app/          Flask API, native window, UI
backend/      Pipeline, QA, OCR, retrieval, LLM, metrics
scripts/      Export, quantise, verify, evaluate, benchmark, package
tests/        50 tests, no weights required
data/samples/ Sample notes and a labelled QA set (answerable, near-miss, off-topic)
results/      Raw output from every script above
```

Full detail is in the [README](../README.md).
