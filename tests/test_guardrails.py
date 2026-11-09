import unittest

from agent.guardrails import VerdictError, extract_json, redact_pii, validate_verdict

RETRIEVED = {"sentiment#labels", "sentiment#rationale-requirements"}


def answer(**changes):
    data = {
        "verdict": "needs_correction",
        "reasons": ["The rationale is too short."],
        "cited_sections": ["sentiment#rationale-requirements"],
        "confidence": 0.8,
    }
    data.update(changes)
    return data


class ValidateVerdictTests(unittest.TestCase):
    def test_valid_answer_passes(self):
        verdict = validate_verdict(answer(), RETRIEVED)
        self.assertEqual(verdict.verdict, "needs_correction")
        self.assertEqual(verdict.cited_sections, ["sentiment#rationale-requirements"])

    def test_accept_may_cite_nothing(self):
        self.assertEqual(validate_verdict(answer(verdict="accept", cited_sections=[]), RETRIEVED).verdict, "accept")

    def test_reject_must_cite_something(self):
        with self.assertRaisesRegex(VerdictError, "at least one"):
            validate_verdict(answer(verdict="reject", cited_sections=[]), RETRIEVED)

    def test_invented_citations_are_rejected(self):
        with self.assertRaisesRegex(VerdictError, "never returned"):
            validate_verdict(answer(cited_sections=["sentiment#made-up"]), RETRIEVED)

    def test_unknown_verdict_is_rejected(self):
        with self.assertRaisesRegex(VerdictError, "verdict"):
            validate_verdict(answer(verdict="maybe"), RETRIEVED)

    def test_extra_and_missing_keys_are_rejected(self):
        with self.assertRaisesRegex(VerdictError, "Unexpected"):
            validate_verdict(answer(note="hi"), RETRIEVED)
        data = answer()
        del data["confidence"]
        with self.assertRaisesRegex(VerdictError, "Missing"):
            validate_verdict(data, RETRIEVED)

    def test_confidence_must_be_a_number_between_zero_and_one(self):
        for bad in (1.5, -0.1, "high", True, None):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(VerdictError, "confidence"):
                    validate_verdict(answer(confidence=bad), RETRIEVED)

    def test_reasons_are_bounded(self):
        for bad in ([], ["x"] * 6, [""], ["a" * 301], "text"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(VerdictError, "reason"):
                    validate_verdict(answer(reasons=bad), RETRIEVED)

    def test_pii_in_reasons_is_masked(self):
        verdict = validate_verdict(answer(reasons=["Mail maria@example.com or call +90 532 123 45 67."]), RETRIEVED)
        self.assertEqual(verdict.reasons, ["Mail [email] or call [phone]."])


class ExtractJsonTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_code_fence_and_prose_are_tolerated(self):
        self.assertEqual(extract_json('Here you go:\n```json\n{"a": 1}\n```\nDone.'), {"a": 1})

    def test_no_json_raises(self):
        with self.assertRaisesRegex(VerdictError, "No JSON"):
            extract_json("I think it is fine.")

    def test_broken_json_raises(self):
        with self.assertRaisesRegex(VerdictError, "Invalid JSON"):
            extract_json('{"a": }')


class RedactTests(unittest.TestCase):
    def test_plain_text_is_unchanged(self):
        self.assertEqual(redact_pii("Nothing sensitive here, 2 hours."), "Nothing sensitive here, 2 hours.")


if __name__ == "__main__":
    unittest.main()
