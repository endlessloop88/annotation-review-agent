"""A rule-based stand-in so the whole pipeline runs without an API key.

This is NOT a language model. It follows fixed rules, calls the same tools through the same loop,
and exists so the project can be demonstrated and tested offline.
"""
import json
import re

from .guardrails import EMAIL, PHONE
from .types import LLMResponse, ToolCall

ALLOWED_LABELS = {"positive", "negative", "neutral", "mixed"}
MIN_RATIONALE_LENGTH = 15
WORD = re.compile(r"[a-zçğıöşü]{4,}")


def _submission(messages: list[dict]) -> dict:
    text = messages[0]["content"]
    return json.loads(text.split("<submission>", 1)[1].split("</submission>", 1)[0])


def _tool_result(messages: list[dict], name: str):
    for message in messages:
        if message["role"] == "tool" and message["name"] == name:
            return json.loads(message["content"])
    return None


class OfflineDemoLLM:
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        submission = _submission(messages)
        search = _tool_result(messages, "search_guidelines")
        stats = _tool_result(messages, "get_annotator_stats")

        if search is None:
            return LLMResponse(tool_calls=[ToolCall("call-1", "search_guidelines", {"query": "labels rationale personal data"})])
        if stats is None:
            return LLMResponse(tool_calls=[ToolCall("call-2", "get_annotator_stats", {"annotator_id": submission["annotator_id"]})])
        return LLMResponse(text=json.dumps(self._decide(submission, search["results"], stats)))

    @staticmethod
    def _section(results: list[dict], keyword: str) -> str | None:
        for result in results:
            if keyword in result["id"]:
                return result["id"]
        return results[0]["id"] if results else None

    def _decide(self, submission: dict, results: list[dict], stats: dict) -> dict:
        label = submission["label"].strip().lower()
        rationale = submission.get("rationale", "") or ""
        text_words = set(WORD.findall(submission["task_text"].lower()))
        rationale_words = set(WORD.findall(rationale.lower()))
        approval = stats.get("approval_rate") if "error" not in stats else None

        verdict, reasons, section = "accept", ["Label, rationale and data handling follow the guidelines."], None
        if label not in ALLOWED_LABELS:
            verdict, section = "reject", self._section(results, "labels")
            reasons = [f"The label '{label}' is not one of: {', '.join(sorted(ALLOWED_LABELS))}."]
        elif EMAIL.search(rationale) or PHONE.search(rationale):
            verdict, section = "needs_correction", self._section(results, "personal-data")
            reasons = ["The rationale contains an e-mail address or phone number. Replace it with [redacted]."]
        elif len(rationale.strip()) < MIN_RATIONALE_LENGTH or not (text_words & rationale_words):
            verdict, section = "needs_correction", self._section(results, "rationale")
            reasons = ["The rationale must be a full sentence that quotes the words that decided the label."]

        if verdict == "accept":
            section = self._section(results, "labels")
        confidence = 0.8 if verdict != "accept" else 0.7
        if approval is not None and approval < 0.5:
            confidence = max(0.4, confidence - 0.2)
            reasons.append("This annotator has a low approval rate, so the review is less certain.")
        cited = [section] if section else [r["id"] for r in results[:1]]
        return {"verdict": verdict, "reasons": reasons, "cited_sections": cited, "confidence": round(confidence, 2)}
