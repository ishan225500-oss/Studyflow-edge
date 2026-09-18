"""API tests with fake models (fast; no weights, no GPU/NPU needed)."""
import io
import time
import unittest

import tests.conftest_path  # noqa: F401
import paths
from engine import StudyEngine
from main import create_app, display_name


class FakeExtractor:
    backend = "easyocr"
    stats = {"pages": 1, "text_layer_pages": 1, "ocr_pages": 0}

    def extract(self, path, progress=None):
        if progress:
            progress("Scanning page 1 of 1")
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    def warmup(self):
        pass


class FakeSummarizer:
    model_name = "fake-t5"

    def summarize(self, text):
        return "Topic line\n- point one\n- point two"

    def warmup(self):
        pass


class FakeQG:
    model_name = "fake-qg"

    def generate(self, text, n=5, style="short"):
        return [{"type": "short", "question": f"Q{i}?", "answer": f"A{i}",
                 "explanation": "", "marks": 1, "options": []} for i in range(n)]

    def warmup(self):
        pass


class FakeQA:
    device = "CPU (fake)"
    runtime = "fake"

    def answer(self, question, context, passages=None):
        found = "mitochondria" in " ".join(passages or [context]).lower()
        return {"answer": "the mitochondria" if found else "", "score": 0.9 if found else 0.01,
                "found": found, "context": "The mitochondria produces ATP.", "engine": "fake QA"}

    def warmup(self):
        pass


class FailingExtractor(FakeExtractor):
    def extract(self, path, progress=None):
        raise ValueError("No readable text found.")


def make_client(extractor=None):
    eng = StudyEngine(extractor=extractor or FakeExtractor(), summarizer=FakeSummarizer(),
                      qgen=FakeQG(), qa=FakeQA(), local_llm=False)
    eng.llm = None
    app = create_app(eng)
    app.config["TESTING"] = True
    return app.test_client()


def wait_job(client, job_id, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").get_json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


NOTES = b"The mitochondria is the powerhouse of the cell and produces ATP through respiration."


class ApiTests(unittest.TestCase):
    def test_index_page_renders(self):
        r = make_client().get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"StudyFlow Edge", r.data)

    def test_status(self):
        s = make_client().get("/api/status").get_json()
        self.assertIn("npu_available", s)
        self.assertFalse(s["has_session"])

    def test_non_ascii_filename_upload_works(self):
        c = make_client()
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "ਨੋਟਸ.txt"), "style": "short"},
                   content_type="multipart/form-data")
        self.assertEqual(r.status_code, 202, r.get_json())
        j = wait_job(c, r.get_json()["job_id"])
        self.assertEqual(j["state"], "done", j)
        self.assertEqual(j["result"]["filename"], "ਨੋਟਸ.txt")
        self.assertEqual(len(j["result"]["questions"]), 5)

    def test_upload_is_deleted_after_processing(self):
        c = make_client()
        before = set(paths.uploads_dir().iterdir())
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "a.txt")},
                   content_type="multipart/form-data")
        wait_job(c, r.get_json()["job_id"])
        self.assertEqual(set(paths.uploads_dir().iterdir()), before)

    def test_bad_extension_rejected_with_json(self):
        r = make_client().post("/api/upload", data={"file": (io.BytesIO(b"x"), "virus.exe")},
                               content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("supported", r.get_json()["error"])

    def test_missing_file(self):
        r = make_client().post("/api/upload", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_processing_error_is_reported_not_hung(self):
        c = make_client(FailingExtractor())
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "a.txt")},
                   content_type="multipart/form-data")
        j = wait_job(c, r.get_json()["job_id"])
        self.assertEqual(j["state"], "error")
        self.assertIn("No readable text", j["error"])

    def test_ask_flow(self):
        c = make_client()
        self.assertEqual(c.post("/api/ask", json={"question": "What?"}).status_code, 400)
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "a.txt")},
                   content_type="multipart/form-data")
        wait_job(c, r.get_json()["job_id"])
        a = c.post("/api/ask", json={"question": "What produces ATP?"}).get_json()
        self.assertTrue(a["found"])
        self.assertEqual(a["answer"], "the mitochondria")
        self.assertEqual(c.post("/api/ask", json={"question": "  "}).status_code, 400)

    def test_regenerate_export_and_clear(self):
        c = make_client()
        r = c.post("/api/upload", data={"file": (io.BytesIO(NOTES), "bio.txt")},
                   content_type="multipart/form-data")
        wait_job(c, r.get_json()["job_id"])
        q = c.post("/api/questions", json={"style": "exam", "num_questions": 3}).get_json()
        self.assertEqual(len(q["questions"]), 3)
        csv = c.get("/api/export/flashcards.csv")
        self.assertEqual(csv.status_code, 200)
        self.assertIn("Q0?", csv.data.decode("utf-8-sig"))
        self.assertTrue(c.post("/api/clear").get_json()["cleared"])
        self.assertFalse(c.get("/api/status").get_json()["has_session"])

    def test_sample_notes(self):
        c = make_client()
        r = c.post("/api/sample", json={"style": "exam"})
        self.assertEqual(r.status_code, 202)
        j = wait_job(c, r.get_json()["job_id"])
        self.assertEqual(j["state"], "done", j)
        self.assertIn("Euler", j["result"]["text_preview"])

    def test_unknown_api_route_returns_json(self):
        r = make_client().get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertIn("error", r.get_json())

    def test_display_name_strips_paths(self):
        self.assertEqual(display_name("C:\\Users\\x\\भौतिकी.pdf"), "भौतिकी.pdf")
        self.assertEqual(display_name(""), "notes")


if __name__ == "__main__":
    unittest.main()
