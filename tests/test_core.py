"""Pure-logic tests: no model weights needed."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import tests.conftest_path  # noqa: F401
from llm import normalise_questions, parse_json_block
from config import SETTINGS
from qa_metrics import summarise, sweep
from qa_module import _sentence_around, best_span
from text_extract import TextExtractor, limit_pixels, render_dpi
from retrieval import BM25Index
from utils import chunk_text, clean_text, split_sentences

NOTES = """Euler's formula states that e^(ix) = cos x + i sin x. It links trigonometry and complex exponentials.
Photosynthesis happens in the chloroplast. Chlorophyll absorbs mostly red and blue light.
The mitochondria is the site of cellular respiration and produces ATP."""


class TextUtils(unittest.TestCase):
    def test_chunks_respect_limit_and_keep_all_text(self):
        long_para = " ".join(["This is sentence number %d." % i for i in range(400)])
        chunks = chunk_text(long_para + "\n" + NOTES, max_chars=500)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertIn("mitochondria", " ".join(chunks))

    def test_huge_single_sentence_is_hard_split(self):
        chunks = chunk_text("x" * 1200, max_chars=500)
        self.assertEqual([len(c) for c in chunks], [500, 500, 200])

    def test_clean_text_joins_hyphenation(self):
        self.assertEqual(clean_text("photo-\nsynthesis  is\n\n\n\nfun"), "photosynthesis is\n\nfun")

    def test_split_sentences(self):
        self.assertEqual(len(split_sentences(NOTES)), 5)


class Retrieval(unittest.TestCase):
    def test_bm25_finds_right_passage(self):
        idx = BM25Index(NOTES, chunk_chars=120)
        top = idx.top_passages("Where is ATP produced?", k=1)
        self.assertIn("mitochondria", top[0])
        top = idx.top_passages("What does Euler's formula say?", k=1)
        self.assertIn("Euler", top[0])

    def test_empty_text(self):
        self.assertEqual(BM25Index("").top_passages("anything"), [])
        self.assertEqual(BM25Index("").top_passages("anything", fallback=False), [])

    def test_no_shared_words_means_no_passage_for_qa(self):
        idx = BM25Index(NOTES, chunk_chars=120)
        self.assertEqual(idx.top_passages("capital of Australia?", fallback=False), [])
        self.assertEqual(len(idx.top_passages("capital of Australia?")), 1)   # calibration keeps a passage


class SpanDecoding(unittest.TestCase):
    def test_best_span_respects_context_mask_and_order(self):
        start = np.array([9, 0, 1, 5, 0, 0], dtype=np.float32)   # token 0 is question: masked
        end = np.array([9, 0, 0, 1, 6, 0], dtype=np.float32)
        mask = np.array([False, True, True, True, True, True])
        st, en, score = best_span(start, end, mask)
        self.assertEqual((st, en), (3, 4))
        self.assertGreater(score, 0.5)

    def test_end_before_start_is_rejected(self):
        start = np.array([0, 0, 0, 8], dtype=np.float32)
        end = np.array([0, 8, 0, 0], dtype=np.float32)
        st, en, _ = best_span(start, end, np.ones(4, bool))
        self.assertLessEqual(st, en)

    def test_max_answer_length(self):
        start = np.zeros(100, np.float32); start[0] = 10
        end = np.zeros(100, np.float32); end[90] = 10
        st, en, _ = best_span(start, end, np.ones(100, bool), max_answer_tokens=30)
        self.assertLess(en - st, 30)

    def test_no_context(self):
        self.assertIsNone(best_span(np.zeros(4), np.zeros(4), np.zeros(4, bool)))


class Evidence(unittest.TestCase):
    def test_first_sentence_keeps_first_character(self):
        text = "Cells are small units of life. Mitochondria make ATP."
        self.assertEqual(_sentence_around(text, 0, 5), "Cells are small units of life.")

    def test_later_sentence(self):
        text = "Cells are small units of life. Mitochondria make ATP. Ribosomes make protein."
        a = text.index("Mitochondria")
        self.assertEqual(_sentence_around(text, a, a + 12), "Mitochondria make ATP.")


class FakePage:
    def __init__(self, text):
        self.text = text
        self.rect = types.SimpleNamespace(width=595, height=842)

    def get_text(self, kind):
        return self.text

    def get_pixmap(self, dpi):
        raise AssertionError("page should not be rendered")


class FakeDoc(list):
    page_count = property(len)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_fitz(pages):
    return types.SimpleNamespace(open=lambda path: FakeDoc(FakePage(t) for t in pages))


class Limits(unittest.TestCase):
    TYPED = "This page has a proper text layer with plenty of characters on it."

    def pdf(self):
        f = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        f.close()
        return f.name

    def test_too_many_pages(self):
        with mock.patch.dict(sys.modules, {"fitz": fake_fitz([self.TYPED] * 5)}), \
                mock.patch.object(SETTINGS, "max_pdf_pages", 4):
            with self.assertRaisesRegex(ValueError, "5 pages"):
                TextExtractor().extract(self.pdf())

    def test_too_many_scanned_pages_refused_before_any_ocr(self):
        pages = [self.TYPED, "", "", ""]
        with mock.patch.dict(sys.modules, {"fitz": fake_fitz(pages)}), \
                mock.patch.object(SETTINGS, "max_ocr_pages", 2):
            with self.assertRaisesRegex(ValueError, "3 scanned pages"):
                TextExtractor().extract(self.pdf())   # FakePage.get_pixmap would fail

    def test_typed_pdf_within_limits(self):
        with mock.patch.dict(sys.modules, {"fitz": fake_fitz([self.TYPED] * 3)}):
            ex = TextExtractor()
            text = ex.extract(self.pdf())
        self.assertIn("text layer", text)
        self.assertEqual(ex.stats, {"pages": 3, "text_layer_pages": 3, "ocr_pages": 0})

    def test_text_length_cap(self):
        f = Path(tempfile.mkdtemp()) / "long.txt"
        f.write_text("word " * 100, encoding="utf-8")
        with mock.patch.object(SETTINGS, "max_text_chars", 50):
            with self.assertRaisesRegex(ValueError, "very long"):
                TextExtractor().extract(str(f))

    def test_render_dpi(self):
        self.assertEqual(render_dpi(595, 842), 200)                   # A4
        dpi = render_dpi(2384, 3370)                                  # A0 poster
        self.assertLess(dpi, 200)
        self.assertLessEqual((2384 / 72 * dpi) * (3370 / 72 * dpi), SETTINGS.max_image_mp * 1e6)

    def test_limit_pixels(self):
        from PIL import Image
        small = Image.new("L", (100, 100))
        self.assertIs(limit_pixels(small), small)
        with mock.patch.object(SETTINGS, "max_image_mp", 0.0025):     # 2,500 px
            out = limit_pixels(Image.new("L", (200, 100)))
        self.assertLessEqual(out.width * out.height, 2500)
        self.assertAlmostEqual(out.width / out.height, 2, delta=0.1)


def row(kind, answers, pred, score, retrieved=True):
    return {"question": "q", "kind": kind, "answers": answers,
            "retrieved": retrieved, "raw_pred": pred, "raw_score": score}


class Metrics(unittest.TestCase):
    ROWS = [
        row("answerable", ["the mitochondria"], "mitochondria", 0.8),
        row("answerable", ["Robert Hooke"], "Robert Hooke", 0.4),
        row("near_miss", [], "newton", 0.3),
        row("near_miss", [], "cork", 0.1),
        row("off_topic", [], "", 0.0, retrieved=False),
    ]

    def test_summary_at_threshold(self):
        s = summarise(self.ROWS, 0.2)
        self.assertEqual(s["answerable"]["exact_match"], 100.0)
        self.assertEqual(s["answerable"]["answered_rate"], 100.0)
        self.assertAlmostEqual(s["unanswerable"]["rejection_rate"], 66.7)
        self.assertEqual(s["unanswerable"]["near_miss_rejection_rate"], 50.0)
        self.assertEqual(s["unanswerable"]["off_topic_rejection_rate"], 100.0)
        self.assertEqual(s["exact_match"], 80.0)       # 4 of 5 correct overall

    def test_sweep_picks_threshold_that_rejects_near_miss(self):
        best = sweep(self.ROWS)["best"]
        self.assertGreater(best["threshold"], 0.3)
        self.assertLessEqual(best["threshold"], 0.4)
        self.assertEqual(best["balanced"], 100.0)


class FakeTorch:
    """Enough of torch to reproduce the thread-local grad rule: a tensor
    produced while grads are on refuses .numpy() until it is detached."""
    def __init__(self):
        self.local = __import__("threading").local()

    class Tensor:
        def __init__(self, arr, requires_grad):
            self.arr, self.requires_grad = arr, requires_grad

        def detach(self):
            return FakeTorch.Tensor(self.arr, False)

        def numpy(self):
            if self.requires_grad:
                raise RuntimeError("Can't call numpy() on Tensor that requires grad. "
                                   "Use tensor.detach().numpy() instead.")
            return self.arr

    @property
    def grad_on(self):
        return getattr(self.local, "on", True)      # on by default in every new thread

    def set_grad_enabled(self, on):
        self.local.on = on

    def from_numpy(self, arr):
        return FakeTorch.Tensor(arr, False)

    def inference_mode(self):
        import contextlib

        @contextlib.contextmanager
        def cm():
            before = self.grad_on
            self.local.on = False
            try:
                yield
            finally:
                self.local.on = before
        return cm()

    def model(self, seq_len):
        def run(input_ids, attention_mask):
            return types.SimpleNamespace(
                start_logits=FakeTorch.Tensor(np.zeros((1, seq_len), dtype=np.float32), self.grad_on),
                end_logits=FakeTorch.Tensor(np.zeros((1, seq_len), dtype=np.float32), self.grad_on))
        return run


class TorchRuntime(unittest.TestCase):
    """The model is loaded on the warm-up thread but used from the web
    server's request threads, where torch still has grads switched on."""
    def run_logits(self, qa, torch):
        out = {}
        def work():
            try:
                out["ok"] = qa._logits(np.ones((1, 8), dtype=np.int64), np.ones((1, 8), dtype=np.int64))
            except Exception as e:                  # noqa: BLE001 - reported below
                out["error"] = e
        t = __import__("threading").Thread(target=work)
        t.start()
        t.join()
        if "error" in out:
            raise out["error"]
        return out["ok"]

    def test_logits_from_another_thread(self):
        from qa_module import QAModule
        torch = FakeTorch()
        torch.set_grad_enabled(False)               # as the loading thread would
        qa = QAModule(runtime="torch")
        qa.runtime = "torch"
        qa._torch_model = torch.model(8)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            start, end = self.run_logits(qa, torch)
        self.assertEqual(start.shape, (1, 8))
        self.assertEqual(end.shape, (1, 8))


class LLMParsing(unittest.TestCase):
    def test_parse_fenced_json(self):
        raw = 'Sure!\n```json\n{"questions": [{"question": "Q?", "answer": "A"}]}\n```'
        self.assertEqual(parse_json_block(raw)["questions"][0]["answer"], "A")

    def test_normalise_mcq_letter_answer(self):
        data = {"questions": [
            {"type": "mcq", "question": "Site of respiration?",
             "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi"], "answer": "B"},
            {"type": "mcq", "question": "Bad", "options": ["x"], "answer": "x"},
            {"type": "long", "question": "Explain photosynthesis.", "answer": "- a\n- b", "marks": "5"},
        ]}
        qs = normalise_questions(data, 5)
        self.assertEqual(len(qs), 2)
        self.assertEqual(qs[0]["answer"], "Mitochondria")
        self.assertEqual(qs[1]["marks"], 5)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            parse_json_block("no json here")


if __name__ == "__main__":
    unittest.main()
