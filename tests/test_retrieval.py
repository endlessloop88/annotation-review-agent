import unittest
from pathlib import Path

from agent.guidelines import load_guidelines, slugify, split_sections
from agent.retrieval import BM25Retriever, tokenize

GUIDELINES = Path(__file__).resolve().parent.parent / "data" / "guidelines"


class GuidelineTests(unittest.TestCase):
    def test_sections_are_split_by_heading_with_stable_ids(self):
        sections = split_sections("demo", "# Title\nignored\n\n## First One\nalpha\n\n## Second\nbeta")
        self.assertEqual([s.id for s in sections], ["demo#first-one", "demo#second"])
        self.assertEqual(sections[0].text, "alpha")

    def test_empty_sections_are_dropped(self):
        self.assertEqual(split_sections("demo", "## Empty\n\n## Full\ntext"), split_sections("demo", "## Full\ntext"))

    def test_slugify_handles_odd_titles(self):
        self.assertEqual(slugify("Sarcasm and irony!"), "sarcasm-and-irony")
        self.assertEqual(slugify("???"), "section")

    def test_shipped_guidelines_load(self):
        ids = {s.id for s in load_guidelines(GUIDELINES)}
        self.assertIn("sentiment#rationale-requirements", ids)
        self.assertIn("sentiment#personal-data", ids)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(load_guidelines(GUIDELINES))

    def test_relevant_section_ranks_first(self):
        top, _ = self.retriever.search("e-mail address phone number")[0]
        self.assertEqual(top.id, "sentiment#personal-data")

    def test_sarcasm_query_finds_sarcasm_section(self):
        top, _ = self.retriever.search("sarcasm irony intended sentiment")[0]
        self.assertEqual(top.id, "sentiment#sarcasm-and-irony")

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(self.retriever.search("quantum chromodynamics"), [])

    def test_k_limits_the_results(self):
        self.assertLessEqual(len(self.retriever.search("label rationale text", k=2)), 2)

    def test_empty_corpus_does_not_crash(self):
        self.assertEqual(BM25Retriever([]).search("anything"), [])

    def test_tokenizer_keeps_accented_letters(self):
        self.assertEqual(tokenize("Çok güzel, değil mi?"), ["çok", "güzel", "değil", "mi"])


if __name__ == "__main__":
    unittest.main()
