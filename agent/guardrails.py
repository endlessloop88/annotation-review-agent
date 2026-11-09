"""Validation of the model's final answer. The model is never trusted to follow the format on its own."""
import json
import re
from dataclasses import dataclass

VERDICTS = ("accept", "needs_correction", "reject")
ALLOWED_KEYS = {"verdict", "reasons", "cited_sections", "confidence"}
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


class VerdictError(ValueError):
    """The answer is not a valid verdict. The message is safe to show back to the model."""


@dataclass(frozen=True)
class Verdict:
    verdict: str
    reasons: list[str]
    cited_sections: list[str]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reasons": self.reasons,
            "cited_sections": self.cited_sections,
            "confidence": self.confidence,
        }


def redact_pii(text: str) -> str:
    """Mask e-mail addresses and phone-like numbers so they never leave the pipeline inside a verdict."""
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text))


def extract_json(text: str) -> dict:
    """Pull one JSON object out of the model's text, tolerating code fences and surrounding prose."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise VerdictError("No JSON object found. Reply with a single JSON object.")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise VerdictError(f"Invalid JSON: {error.msg}.") from error
    if not isinstance(data, dict):
        raise VerdictError("The answer must be a JSON object.")
    return data


def validate_verdict(data: dict, retrieved_ids: set[str]) -> Verdict:
    unknown = set(data) - ALLOWED_KEYS
    if unknown:
        raise VerdictError(f"Unexpected keys: {', '.join(sorted(unknown))}.")
    missing = ALLOWED_KEYS - set(data)
    if missing:
        raise VerdictError(f"Missing keys: {', '.join(sorted(missing))}.")

    verdict = data["verdict"]
    if verdict not in VERDICTS:
        raise VerdictError(f"'verdict' must be one of {', '.join(VERDICTS)}.")

    reasons = data["reasons"]
    if not isinstance(reasons, list) or not 1 <= len(reasons) <= 5 or not all(isinstance(r, str) and r.strip() for r in reasons):
        raise VerdictError("'reasons' must be a list of 1 to 5 non-empty strings.")
    if any(len(reason) > 300 for reason in reasons):
        raise VerdictError("Each reason must be at most 300 characters.")

    cited = data["cited_sections"]
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        raise VerdictError("'cited_sections' must be a list of section ids.")
    invented = [c for c in cited if c not in retrieved_ids]
    if invented:
        raise VerdictError(
            f"These sections were never returned by search_guidelines: {', '.join(invented)}. Cite only sections you retrieved."
        )
    if verdict != "accept" and not cited:
        raise VerdictError("A needs_correction or reject verdict must cite at least one guideline section.")

    confidence = data["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise VerdictError("'confidence' must be a number between 0 and 1.")

    return Verdict(
        verdict=verdict,
        reasons=[redact_pii(reason.strip()) for reason in reasons],
        cited_sections=list(cited),
        confidence=float(confidence),
    )
