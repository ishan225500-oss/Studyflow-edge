"""
build_desktop.py
Package StudyFlow Edge into a Windows app folder (dist/StudyFlowEdge/) and a
zip ready to attach to a GitHub Release.

Run ON Windows, with the Python you want to ship:
  - native ARM64 Python  -> native Snapdragon app (recommended when all
    dependencies you use provide win-arm64 wheels)
  - x64 Python           -> x64 app that runs under emulation on Snapdragon

    python scripts/download_models.py          # so models can be bundled
    python scripts/build_desktop.py --bundle-models

PyInstaller does not cross-compile.
"""
import argparse
import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEP = ";" if sys.platform.startswith("win") else ":"

COLLECT_DATA = ["easyocr", "transformers"]
COLLECT_BINARIES = ["onnxruntime", "llama_cpp"]      # QnnHtp.dll etc. live in onnxruntime/capi
COPY_METADATA = ["transformers", "tokenizers", "huggingface_hub", "safetensors", "tqdm", "regex",
                 "requests", "packaging", "filelock", "numpy", "pyyaml", "torch"]


def installed(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-models", action="store_true",
                    help="ship models/ inside the app (fully offline, larger download)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    plat = sysconfig.get_platform()
    print(f"Building with Python {sys.version.split()[0]} for {plat}")
    if not sys.platform.startswith("win"):
        print("Note: this build will only run on the OS you build it on.")

    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--name", "StudyFlowEdge", "--windowed",
        "--paths", str(ROOT / "backend"), "--paths", str(ROOT / "app"),
        "--add-data", f"{ROOT / 'app' / 'templates'}{SEP}app/templates",
        "--add-data", f"{ROOT / 'data' / 'samples'}{SEP}data/samples",
    ]
    for m in ("engine", "text_extract", "qa_module", "summariser", "question_gen", "llm", "retrieval"):
        cmd += ["--hidden-import", m]
    for pkg in COLLECT_DATA:
        if installed(pkg):
            cmd += ["--collect-data", pkg]
    for pkg in COLLECT_BINARIES:
        if installed(pkg):
            cmd += ["--collect-binaries", pkg]
    for dist in COPY_METADATA:
        if installed(dist.replace("-", "_").replace("pyyaml", "yaml")):
            cmd += ["--copy-metadata", dist]

    if args.bundle_models:
        models = ROOT / "models"
        if not any(p.name != ".gitkeep" for p in models.iterdir()):
            raise SystemExit("models/ is empty. Run scripts/download_models.py first.")
        cmd += ["--add-data", f"{models}{SEP}models"]

    cmd.append(str(ROOT / "app" / "desktop.py"))
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    dist_dir = ROOT / "dist" / "StudyFlowEdge"
    print(f"\nBuilt {dist_dir / 'StudyFlowEdge.exe'}")
    if not args.no_zip:
        arch = plat.replace("win-", "windows-")
        archive = shutil.make_archive(str(ROOT / "dist" / f"StudyFlowEdge-{arch}"), "zip", dist_dir)
        print(f"Release zip: {archive}")
    print("Test it on a clean machine (or a fresh Windows user account) before submitting.")


if __name__ == "__main__":
    main()
