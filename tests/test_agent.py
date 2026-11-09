import json
import unittest
from pathlib import Path

from agent.core import ReviewFailed, review
from agent.guidelines import load_guidelines
from agent.retrieval import BM25Retriever
from agent.tools import ToolContext, load_stats, run_tool
from agent.types import LLMResponse, ToolCall

DATA = Path(__file__).resolve().parent.parent / "data"
SUBMISSION = {
    "id": "t-1",
    "annotator_id": "ann-001",
    "task_text": "The delivery was late but the staff were kind.",
    "label": "mixed",
    "rationale": "Late delivery is negative but kind staff are positive.",
}
GOOD_ANSWER = json.dumps(
    {
        "verdict": "needs_correction",
        "reasons": ["Needs work."],
        "cited_sections": ["sentiment#labels"],
        "confidence": 0.6,
    }
)


class ScriptedLLM:
    """Replays prepared responses and records what the agent sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen_messages = []

    def complete(self, system, messages, tools):
        self.seen_messages.append(list(messages))
        return self.responses.pop(0)


def search_call(query="labels"):
    return LLMResponse(tool_calls=[ToolCall("c1", "search_guidelines", {"query": query})])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.context = ToolContext(
            retriever=BM25Retriever(load_guidelines(DATA / "guidelines")), stats=load_stats(DATA / "annotator_stats.json")
        )

    def test_tool_call_then_valid_verdict(self):
        llm = ScriptedLLM([search_call(), LLMResponse(text=GOOD_ANSWER)])
        verdict, trace = review(SUBMISSION, llm, self.context)
        self.assertEqual(verdict.verdict, "needs_correction")
        self.assertEqual(trace, [{"tool": "search_guidelines", "arguments": {"query": "labels"}}])

    def test_tool_result_is_sent_back_to_the_model(self):
        llm = ScriptedLLM([search_call(), LLMResponse(text=GOOD_ANSWER)])
        review(SUBMISSION, llm, self.context)
        last_turn = llm.seen_messages[-1]
        self.assertEqual(last_turn[-1]["role"], "tool")
        self.assertIn("sentiment#labels", last_turn[-1]["content"])

    def test_invalid_answer_is_repaired(self):
        llm = ScriptedLLM([search_call(), LLMResponse(text="Looks fine to me."), LLMResponse(text=GOOD_ANSWER)])
        verdict, _ = review(SUBMISSION, llm, self.context)
        self.assertEqual(verdict.verdict, "needs_correction")
        repair_request = llm.seen_messages[-1][-1]
        self.assertEqual(repair_request["role"], "user")
        self.assertIn("not valid", repair_request["content"])

    def test_citing_an_unretrieved_section_is_never_accepted(self):
        invented = json.dumps(
            {"verdict": "reject", "reasons": ["Bad."], "cited_sections": ["sentiment#made-up"], "confidence": 0.9}
        )
        llm = ScriptedLLM([search_call(), LLMResponse(text=invented), LLMResponse(text=invented), LLMResponse(text=invented)])
        with self.assertRaises(ReviewFailed):
            review(SUBMISSION, llm, self.context)

    def test_gives_up_after_max_steps(self):
        llm = ScriptedLLM([search_call() for _ in range(10)])
        with self.assertRaisesRegex(ReviewFailed, "steps"):
            review(SUBMISSION, llm, self.context, max_steps=3)

    def test_each_review_starts_with_no_retrieved_sections(self):
        llm = ScriptedLLM([search_call(), LLMResponse(text=GOOD_ANSWER)])
        review(SUBMISSION, llm, self.context)
        self.assertTrue(self.context.retrieved_ids)
        llm = ScriptedLLM([LLMResponse(text=GOOD_ANSWER) for _ in range(3)])  # cites a section but never searched in this review
        with self.assertRaises(ReviewFailed):
            review(SUBMISSION, llm, self.context)

    def test_submission_is_validated_before_calling_the_model(self):
        llm = ScriptedLLM([])
        with self.assertRaisesRegex(ValueError, "missing fields"):
            review({"id": "x"}, llm, self.context)
        with self.assertRaisesRegex(ValueError, "longer than"):
            review({**SUBMISSION, "rationale": "a" * 4001}, llm, self.context)

    def test_submission_is_wrapped_as_untrusted_data(self):
        llm = ScriptedLLM([search_call(), LLMResponse(text=GOOD_ANSWER)])
        review(SUBMISSION, llm, self.context)
        first = llm.seen_messages[0][0]["content"]
        self.assertTrue(first.startswith("<submission>") and first.rstrip().endswith("</submission>"))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.context = ToolContext(
            retriever=BM25Retriever(load_guidelines(DATA / "guidelines")), stats=load_stats(DATA / "annotator_stats.json")
        )

    def test_search_records_what_the_model_saw(self):
        result = json.loads(run_tool("search_guidelines", {"query": "personal data e-mail"}, self.context))
        self.assertEqual(result["results"][0]["id"], "sentiment#personal-data")
        self.assertIn("sentiment#personal-data", self.context.retrieved_ids)

    def test_stats_for_known_and_unknown_annotators(self):
        known = json.loads(run_tool("get_annotator_stats", {"annotator_id": "ann-002"}, self.context))
        self.assertEqual(known["approval_rate"], 0.42)
        unknown = json.loads(run_tool("get_annotator_stats", {"annotator_id": "nobody"}, self.context))
        self.assertIn("unknown annotator", unknown["error"])

    def test_bad_arguments_and_unknown_tools_return_errors(self):
        for name, arguments in (("search_guidelines", {}), ("search_guidelines", {"query": 5}), ("get_annotator_stats", {"annotator_id": 1}), ("nope", {})):
            with self.subTest(name=name, arguments=arguments):
                self.assertIn("error", json.loads(run_tool(name, arguments, self.context)))


if __name__ == "__main__":
    unittest.main()
