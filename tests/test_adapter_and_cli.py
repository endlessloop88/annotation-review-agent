import io
import json
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from agent.anthropic_llm import AnthropicLLM
from agent.cli import main


class FakeClient:
    """Mimics the shape of the SDK's response so the adapter can be tested without a network."""

    def __init__(self, content):
        self.calls = []
        self.messages = self
        self._content = content

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._content)


TOOLS = [{"name": "search_guidelines", "description": "d", "parameters": {"type": "object", "properties": {}}}]


class AdapterTests(unittest.TestCase):
    def test_reads_text_and_tool_use_blocks(self):
        content = [
            SimpleNamespace(type="text", text="Searching."),
            SimpleNamespace(type="tool_use", id="tu_1", name="search_guidelines", input={"query": "labels"}),
        ]
        client = FakeClient(content)
        response = AnthropicLLM(model="m", client=client).complete("sys", [{"role": "user", "content": "hi"}], TOOLS)
        self.assertEqual(response.text, "Searching.")
        self.assertEqual((response.tool_calls[0].id, response.tool_calls[0].arguments), ("tu_1", {"query": "labels"}))
        sent = client.calls[0]
        self.assertEqual(sent["tools"][0]["input_schema"], TOOLS[0]["parameters"])
        self.assertEqual(sent["system"], "sys")

    def test_neutral_messages_become_provider_messages(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "text": "", "tool_calls": [{"id": "a", "name": "search_guidelines", "arguments": {"query": "x"}}, {"id": "b", "name": "get_annotator_stats", "arguments": {"annotator_id": "u"}}]},
            {"role": "tool", "tool_call_id": "a", "name": "search_guidelines", "content": "{}"},
            {"role": "tool", "tool_call_id": "b", "name": "get_annotator_stats", "content": "{}"},
            {"role": "assistant", "text": "not json", "tool_calls": []},
            {"role": "user", "content": "fix it"},
        ]
        converted = AnthropicLLM.to_provider_messages(messages)
        self.assertEqual([m["role"] for m in converted], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual([b["type"] for b in converted[1]["content"]], ["tool_use", "tool_use"])
        self.assertEqual([b["tool_use_id"] for b in converted[2]["content"]], ["a", "b"])  # one user turn, two results

    def test_default_model_can_come_from_the_environment(self):
        import os

        os.environ["ANTHROPIC_MODEL"] = "from-env"
        try:
            self.assertEqual(AnthropicLLM(client=FakeClient([])).model, "from-env")
        finally:
            del os.environ["ANTHROPIC_MODEL"]


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(list(args))
        return code, [json.loads(line) for line in buffer.getvalue().splitlines()]

    def test_demo_reviews_all_sample_submissions(self):
        code, entries = self.run_cli()
        self.assertEqual(code, 0)
        verdicts = {entry["id"]: entry["verdict"]["verdict"] for entry in entries}
        self.assertEqual(
            verdicts,
            {"s-001": "accept", "s-002": "needs_correction", "s-003": "reject", "s-004": "needs_correction", "s-005": "needs_correction"},
        )

    def test_prompt_injection_in_a_rationale_does_not_change_the_outcome(self):
        _, entries = self.run_cli()
        injected = next(entry for entry in entries if entry["id"] == "s-005")
        self.assertNotEqual(injected["verdict"]["verdict"], "accept")

    def test_trace_flag_lists_the_tool_calls(self):
        _, entries = self.run_cli("--trace")
        self.assertEqual([call["tool"] for call in entries[0]["tool_calls"]], ["search_guidelines", "get_annotator_stats"])

    def test_every_verdict_cites_a_real_section(self):
        _, entries = self.run_cli()
        for entry in entries:
            self.assertTrue(entry["verdict"]["cited_sections"][0].startswith("sentiment#"))


if __name__ == "__main__":
    unittest.main()
