# Submission checklist (deadline: 30 Sep 2026, 11:59 PM IST)

Judging: Technical Implementation, Application Use Case & Innovation,
Deployment & Accessibility, Presentation & Documentation (100 points total).

## Before 22 Sep: evidence for Technical Implementation
- [ ] `python -m unittest discover -s tests -t .` passes on your machine
- [ ] `python scripts/download_models.py --llm llama3.2-3b --handwriting`
- [ ] `python scripts/export_qa_onnx.py` (parity check passes)
- [ ] `python scripts/quantize_qa_qnn.py`
- [ ] `python scripts/evaluate_qa.py --compare` (QDQ accuracy close to FP32; if not, retry with `--calib-text` on more notes)
- [ ] Add 10-20 questions from your own lecture notes to a copy of `data/samples/qa_eval.json` (including near-miss ones the notes don't answer)
- [ ] `python scripts/evaluate_qa.py --onnx models/qa/model.qdq.onnx --sweep` on both sets; set `STUDYFLOW_QA_MIN_SCORE` default in `backend/config.py` to the chosen value and record the rejection rate
- [ ] `python scripts/aihub_profile.py --verify` (save the job links; screenshot the AI Hub profile page)
- [ ] If you can access a Snapdragon PC: `python scripts/check_npu.py`
- [ ] `python scripts/benchmark.py --doc <a real 10-page lecture PDF> --question "<a question that PDF answers>"`
- [ ] Copy the numbers into the README "Measured results" table. Leave "not measured" where true.

## Before 25 Sep: Deployment & Accessibility
- [ ] `python scripts/build_desktop.py --bundle-models`
- [ ] Run the built app on a machine/user account with no Python and no internet
- [ ] Publish a GitHub Release with the zip; link it at the top of the README
- [ ] Try the app using only the keyboard, at 150% text size, and in dark mode

## Before 29 Sep: Presentation & Documentation
- [ ] Add docs/screenshot.png from a real run (real models) and enable the image line at the top of the README
- [ ] 2-3 minute demo video: problem (20s) -> sample notes -> handwritten page -> exam mode -> ask a question it can't answer -> NPU evidence (AI Hub profile / check_npu output) -> offline (Wi-Fi off)
- [ ] 6-8 slide deck, one section per judging criterion, same numbers as the README
- [ ] 1-2 page project description
- [ ] Clean clone of the repo: install steps work exactly as written
- [ ] Submit on Unstop by 29 Sep; keep 30 Sep as buffer
