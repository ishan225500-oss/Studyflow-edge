"""
desktop.py
StudyFlow Edge as a native Windows app: the Flask server runs on a free
localhost port in a background thread and the UI opens in a WebView2
window via pywebview. This is the entry point PyInstaller packages into
StudyFlowEdge.exe (scripts/build_desktop.py).
"""
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import main as server  # noqa: E402

HOST = "127.0.0.1"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, 0))
        return s.getsockname()[1]


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.2)
    return False


def main():
    import webview

    port = free_port()
    threading.Thread(target=server.run, kwargs={"port": port}, daemon=True).start()
    url = f"http://{HOST}:{port}/"
    if not wait_for_server(url + "api/status"):
        print("StudyFlow Edge server did not start.", file=sys.stderr)
        sys.exit(1)

    try:  # lets "Export flashcards" save a CSV (pywebview >= 5)
        webview.settings["ALLOW_DOWNLOADS"] = True
    except Exception:
        pass

    webview.create_window("StudyFlow Edge", url, width=1180, height=820,
                          min_size=(720, 560), text_select=True)
    webview.start()


if __name__ == "__main__":
    main()
