"""Grounded answering: the QA span is the answer; the LLM may only explain it."""
import unittest

import tests.conftest_path  # noqa: F401
from engine import StudyEngine
from llm import NOT_FOUND, clean_explanation
from tests.test_app import FakeExtractor, FakeQG, FakeSummarizer

NOTES = ("The mitochondria is the powerhouse of the cell and produces ATP through respiration.\n"
         "Ribosomes are the site of protein synthesis.")


class CountingQA:
    """Finds 'the mitochondria' only when a passage mentions mitochondria."""
    device = "CPU (fake)"
    runtime = "fake"

    def __init__(self):
        self.calls = 0

    def answer(self, question, context, passages=None):
        self.calls += 1
        found = "mitochondria" in " ".join(passages or [context]).lower()
        return {"answer": "the mitochondria" if found else "some guess",
                "score": 0.9 if found else 0.02, "found": found,
                "context": "The mitochondria is the powerhouse of the cell and produces ATP.",
                "engine": "fake QA"}

    def warmup(self):
        pass


class FakeLLM:
    name = "fake-llama"

    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []

    def explain(self, question, answer, evidence, passages):
        self.calls.append({"question": question, "answer": answer,
                           "evidence": evidence, "passages": passages})
        if self.error:
            raise self.error
        return clean_explanation(self.reply)

    def warmup(self):
        pass


def engine(llm=None, qa=None):
    eng = StudyEngine(extractor=FakeExtractor(), summarizer=FakeSummarizer(), qgen=FakeQG(),
                      qa=qa or CountingQA(), local_llm=llm if llm is not None else False)
    if llm is None:
        eng.llm = None
    return eng


class Grounding(unittest.TestCase):
    def test_llm_explains_but_cannot_replace_the_answer(self):
        llm = FakeLLM(reply="Actually the nucleus makes ATP.")   # a would-be hallucination
        r = engine(llm).ask("What produces ATP?", NOTES, None)
        self.assertTrue(r["found"])
        self.assertEqual(r["answer"], "the mitochondria")
        self.assertIn("mitochondria", r["evidence"])
        self.assertEqual(r["explanation"], "Actually the nucleus makes ATP.")
        self.assertIn("fake-llama", r["explanation_engine"])
        self.assertEqual(r["engine"], "fake QA")
        # the LLM is told the verified answer and the quoted line
        self.assertEqual(llm.calls[0]["answer"], "the mitochondria")
        self.assertIn("powerhouse", llm.calls[0]["evidence"])

    def test_llm_not_asked_when_qa_is_unsure(self):
        llm = FakeLLM(reply="Ribosomes make proteins.")
        notes = "Ribosomes are the site of protein synthesis. Proteins are made of amino acids."
        r = engine(llm).ask("Where does protein synthesis happen?", notes, None)
        self.assertFalse(r["found"])
        self.assertEqual((r["answer"], r["evidence"], r["explanation"]), ("", "", ""))
        self.assertEqual(r["reason"], "low_confidence")
        self.assertEqual(llm.calls, [])

    def test_off_topic_question_never_reaches_the_models(self):
        qa, llm = CountingQA(), FakeLLM(reply="Canberra.")
        r = engine(llm, qa).ask("What is the capital of Australia?", NOTES, None)
        self.assertFalse(r["found"])
        self.assertEqual(r["reason"], "no_matching_notes")
        self.assertEqual(qa.calls, 0)
        self.assertEqual(llm.calls, [])

    def test_llm_declining_keeps_answer_without_explanation(self):
        r = engine(FakeLLM(reply=f"{NOT_FOUND}")).ask("What produces ATP?", NOTES, None)
        self.assertTrue(r["found"])
        self.assertEqual(r["answer"], "the mitochondria")
        self.assertEqual(r["explanation"], "")
        self.assertIsNone(r["explanation_engine"])

    def test_llm_failure_keeps_answer(self):
        r = engine(FakeLLM(error=RuntimeError("out of memory"))).ask("What produces ATP?", NOTES, None)
        self.assertTrue(r["found"])
        self.assertEqual(r["explanation"], "")

    def test_without_llm(self):
        r = engine().ask("What produces ATP?", NOTES, None)
        self.assertEqual((r["answer"], r["explanation"]), ("the mitochondria", ""))

    def test_clean_explanation(self):
        self.assertEqual(clean_explanation("  Because of respiration. "), "Because of respiration.")
        self.assertEqual(clean_explanation(""), "")
        self.assertEqual(clean_explanation(f"Sorry, {NOT_FOUND}."), "")


if __name__ == "__main__":
    unittest.main()
