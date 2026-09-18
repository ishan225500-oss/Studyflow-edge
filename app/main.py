"""
main.py
StudyFlow Edge local server (Flask). Serves the UI and a small JSON API.
Binds to 127.0.0.1 only; nothing leaves the machine.

Upload flow: POST /api/upload returns a job id immediately; the pipeline
runs in a background thread and the UI polls GET /api/jobs/<id> for
progress, so long PDFs never freeze the window.

Dev mode:  python app/main.py   ->  http://127.0.0.1:5000
"""
import csv
import io
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from flask import Flask, Response, jsonify, render_template, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

import paths  # noqa: E402
from config import SETTINGS  # noqa: E402
from text_extract import SUPPORTED_EXTENSIONS  # noqa: E402
from utils import get_logger  # noqa: E402

log = get_logger("app")

VALID_STYLES = {"mcq", "short", "exam"}
VALID_OCR = {"easyocr", "handwriting"}


def display_name(raw: str) -> str:
    """Keep the user's filename (any script, e.g. Punjabi/Hindi) for display only.
    The file is stored under a random name, so this never touches the filesystem."""
    name = Path((raw or "").replace("\\", "/")).name.strip()
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:120] or "notes"


LOCAL_HOSTNAMES = {"127.0.0.1", "localhost", "[::1]"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def host_without_port(host: str) -> str:
    host = (host or "").strip().lower()
    if host.startswith("["):                      # [::1]:5000
        return host[: host.find("]") + 1]
    return host.rsplit(":", 1)[0]


def request_blocked_reason(method: str, host: str, origin, fetch_site) -> str:
    """Why a request must be refused, or "" if it's fine.

    The server listens on 127.0.0.1, but any web page open in the student's
    browser can still send requests to it. Two checks keep the notes private:
    - Host must be a loopback name. This defeats DNS rebinding, where an
      attacker's domain is re-pointed at 127.0.0.1 to read /api/status.
    - Requests that change state must come from this app's own page, so other
      sites can't upload files, start OCR or clear the session.
    """
    if host_without_port(host) not in LOCAL_HOSTNAMES:
        return "host"
    if method in SAFE_METHODS:
        return ""
    if fetch_site in ("cross-site", "same-site"):   # same-site = other localhost port
        return "cross-site"
    if origin is not None and origin != f"http://{host}":
        return "origin"
    return ""


def create_app(engine=None) -> Flask:
    app = Flask(__name__, template_folder=str(paths.templates_dir()))
    app.config["MAX_CONTENT_LENGTH"] = SETTINGS.max_upload_mb * 1024 * 1024

    if engine is None:
        from engine import StudyEngine
        engine = StudyEngine()
    app.engine = engine

    state = {"session": None}          # single local user
    jobs = {}
    jobs_lock = threading.Lock()

    def public_session(sess):
        if not sess:
            return None
        return {k: sess[k] for k in ("filename", "summary", "questions", "stats", "engines",
                                     "timings", "text_preview")}

    # ---------- local-only guard ----------
    @app.before_request
    def local_only():
        why = request_blocked_reason(request.method, request.host,
                                     request.headers.get("Origin"),
                                     request.headers.get("Sec-Fetch-Site"))
        if why:
            log.warning(f"Refused {request.method} {request.path} ({why})")
            return jsonify({"error": "StudyFlow Edge only accepts requests from its own window."}), 403

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    # ---------- errors: always JSON under /api ----------
    @app.errorhandler(Exception)
    def handle_error(e):
        if isinstance(e, HTTPException):
            code, msg = e.code, e.description
            if code == 413:
                msg = f"File is larger than {SETTINGS.max_upload_mb} MB. Split it and upload the parts."
        else:
            log.exception("Unhandled error")
            code, msg = 500, f"Something went wrong: {type(e).__name__}: {e}"
        if request.path.startswith("/api/"):
            return jsonify({"error": msg}), code
        return (msg, code) if isinstance(e, HTTPException) else (msg, 500)

    # ---------- pages ----------
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        providers = []
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
        except Exception:
            pass
        return jsonify({
            **app.engine.status(),
            "npu_available": "QNNExecutionProvider" in providers,
            "onnx_providers": providers,
            "has_session": state["session"] is not None,
            "session": public_session(state["session"]),
            "keep_uploads": SETTINGS.keep_uploads,
        })

    # ---------- upload + jobs ----------
    @app.post("/api/upload")
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "Choose a PDF, image or text file first."}), 400
        name = display_name(f.filename)
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return jsonify({"error": f"'{ext or 'no extension'}' files aren't supported. "
                                     "Use PDF, PNG, JPG or TXT."}), 400
        try:
            n = max(1, min(15, int(request.form.get("num_questions", 5))))
        except ValueError:
            n = 5
        style = request.form.get("style", "mcq")
        style = style if style in VALID_STYLES else "mcq"
        ocr = request.form.get("ocr", SETTINGS.ocr_backend)
        ocr = ocr if ocr in VALID_OCR else "easyocr"

        stored = paths.uploads_dir() / f"{uuid.uuid4().hex}{ext}"
        f.save(stored)
        return jsonify({"job_id": start_job(stored, name, n, style, ocr)}), 202

    @app.post("/api/sample")
    def sample():
        src = paths.resource_dir() / "data" / "samples" / "sample_notes.txt"
        if not src.is_file():
            return jsonify({"error": "Sample notes are missing from this install."}), 404
        data = request.get_json(silent=True) or {}
        style = data.get("style", "mcq")
        style = style if style in VALID_STYLES else "mcq"
        stored = paths.uploads_dir() / f"{uuid.uuid4().hex}.txt"
        shutil.copyfile(src, stored)
        return jsonify({"job_id": start_job(stored, "Sample notes (maths, biology, physics).txt",
                                            5, style, "easyocr")}), 202

    def start_job(stored: Path, name: str, n: int, style: str, ocr: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "state": "running", "stage": "queued",
               "message": "Starting", "error": None, "started": time.time()}
        with jobs_lock:
            jobs[job_id] = job

        def progress(stage, message):
            job.update(stage=stage, message=message)

        def run():
            try:
                res = app.engine.process(str(stored), num_questions=n, style=style,
                                         ocr_backend=ocr, progress=progress)
                res["filename"] = name
                res["style"] = style
                res["text_preview"] = res["text"][:600]
                state["session"] = res
                job.update(state="done", stage="done", message="Ready")
            except Exception as e:
                log.exception("Processing failed")
                job.update(state="error", error=str(e) or type(e).__name__)
            finally:
                job["elapsed_s"] = round(time.time() - job["started"], 2)
                if not SETTINGS.keep_uploads:
                    stored.unlink(missing_ok=True)

        threading.Thread(target=run, daemon=True).start()
        return job_id

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Unknown job. Upload the file again."}), 404
        out = {k: job.get(k) for k in ("id", "state", "stage", "message", "error", "elapsed_s")}
        if job["state"] == "done":
            out["result"] = public_session(state["session"])
        return jsonify(out)

    # ---------- study actions ----------
    def require_session():
        if not state["session"]:
            return None, (jsonify({"error": "Upload your notes first."}), 400)
        return state["session"], None

    @app.post("/api/questions")
    def regenerate():
        sess, err = require_session()
        if err:
            return err
        data = request.get_json(silent=True) or {}
        style = data.get("style", "mcq")
        style = style if style in VALID_STYLES else "mcq"
        try:
            n = max(1, min(15, int(data.get("num_questions", 5))))
        except (TypeError, ValueError):
            n = 5
        res = app.engine.questions_only(sess["text"], n, style)
        sess["questions"] = res["questions"]
        sess["style"] = style
        sess["engines"]["questions"] = res["engine"]
        return jsonify(res)

    @app.post("/api/ask")
    def ask():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Type a question first."}), 400
        if len(question) > 500:
            return jsonify({"error": "Keep questions under 500 characters."}), 400
        sess, err = require_session()
        if err:
            return err
        return jsonify(app.engine.ask(question, sess["text"], sess.get("index")))

    @app.post("/api/clear")
    def clear():
        state["session"] = None
        with jobs_lock:
            jobs.clear()
        return jsonify({"cleared": True})

    @app.get("/api/export/flashcards.csv")
    def export_csv():
        sess, err = require_session()
        if err:
            return err
        buf = io.StringIO()
        w = csv.writer(buf)
        for q in sess["questions"]:
            front = q["question"]
            if q.get("options"):
                front += "\n" + "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(q["options"]))
            w.writerow([front, q["answer"]])
        stem = Path(sess["filename"]).stem.encode("ascii", "ignore").decode() or "notes"
        return Response(
            "\ufeff" + buf.getvalue(),  # BOM so Excel opens UTF-8 correctly
            mimetype="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{stem}-flashcards.csv"'},
        )

    return app


def run(port: int = 5000, debug: bool = False, engine=None, warmup: bool = True):
    app = create_app(engine)
    if warmup:
        threading.Thread(target=app.engine.warmup, daemon=True).start()
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    run(port=args.port, debug=args.debug)
