"""
download_models.py
Fetch every model StudyFlow Edge needs into models/ so the app runs fully
offline afterwards (and so build_desktop.py --bundle-models can ship them).

    python scripts/download_models.py                    # QA + OCR + compact models
    python scripts/download_models.py --llm llama3.2-3b  # + local LLM (~2 GB)
    python scripts/download_models.py --handwriting      # + TrOCR handwritten

Requires internet once. Needs: huggingface_hub, easyocr.
"""
import argparse
import os

os.environ["HF_HUB_OFFLINE"] = "0"          # config.py would otherwise force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import _common  # noqa: E402,F401
import config  # noqa: E402  (sets HF_HOME -> models/hf)
import paths  # noqa: E402
from config import SETTINGS  # noqa: E402

LLMS = {
    "llama3.2-3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "llama3.2-1b": ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "qwen2.5-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
}

SKIP = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*.tflite", "coreml/*", "onnx/*", "*.mlmodel"]


def fetch_hf(repo: str):
    from huggingface_hub import HfApi, snapshot_download
    files = HfApi().list_repo_files(repo)
    ignore = list(SKIP)
    if any(f.endswith(".safetensors") for f in files):
        ignore.append("*.bin")   # don't download the same weights twice
    print(f"-> {repo}")
    snapshot_download(repo, ignore_patterns=ignore)


def fetch_easyocr(langs):
    import easyocr
    store = paths.models_dir() / "easyocr"
    store.mkdir(parents=True, exist_ok=True)
    print(f"-> EasyOCR {langs} into {store}")
    easyocr.Reader(langs, gpu=False, model_storage_directory=str(store),
                   download_enabled=True, verbose=False)


def fetch_llm(key: str):
    from huggingface_hub import hf_hub_download
    repo, filename = LLMS[key]
    dest = paths.models_dir() / "llm"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"-> {repo}/{filename} into {dest}")
    hf_hub_download(repo, filename, local_dir=str(dest))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=list(LLMS) + ["none"], default="none")
    ap.add_argument("--handwriting", action="store_true", help="also fetch TrOCR (handwritten)")
    ap.add_argument("--no-compact", action="store_true", help="skip T5 summary/QG fallback models")
    args = ap.parse_args()

    print(f"Models folder: {paths.models_dir()}")
    fetch_hf(SETTINGS.qa_model)
    if not args.no_compact:
        fetch_hf(SETTINGS.sum_model)
        fetch_hf(SETTINGS.qg_model)
    if args.handwriting:
        fetch_hf(SETTINGS.trocr_model)
    fetch_easyocr(SETTINGS.ocr_languages)
    if args.llm != "none":
        fetch_llm(args.llm)
    print("\nDone. The app will now start without internet access.")


if __name__ == "__main__":
    main()
