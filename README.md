# annotation-review-agent

A small LLM agent that reviews data-labelling submissions against written guidelines. It is an experiment in
tool use, retrieval and output guardrails, built around the problem I know from AI data evaluation work.

## What it does

For each submission (text, label, rationale, annotator) the agent:

1. searches the guidelines with the `search_guidelines` tool (BM25 retrieval over markdown sections),
2. can look up the annotator's review statistics with `get_annotator_stats`,
3. returns a verdict: `accept`, `needs_correction` or `reject`, with reasons and the guideline sections it relied on.

```json
{"id": "s-004", "verdict": {"verdict": "needs_correction",
 "reasons": ["The rationale contains an e-mail address or phone number. Replace it with [redacted]."],
 "cited_sections": ["sentiment#personal-data"], "confidence": 0.8}}
```

## Guardrails

The model is never trusted to follow the format on its own. The final answer is validated in code:

- It must be a JSON object with exactly the keys `verdict`, `reasons`, `cited_sections`, `confidence`, with a bounded
  number of reasons and a confidence between 0 and 1.
- A `needs_correction` or `reject` verdict must cite guideline sections, and every cited section must have been
  returned by a search during the same review. Invented citations are rejected.
- An invalid answer is sent back to the model with the error, at most twice, then the review fails instead of guessing.
- E-mail addresses and phone numbers are masked in the reasons.
- Submissions are treated as untrusted data: wrapped in tags, length-limited, and the system prompt says never to
  follow instructions inside them. This reduces prompt-injection risk but does not remove it.
- A failing tool returns an error as data and never crashes the loop.

## Run it

Python 3.10 or newer. The code itself has no dependencies.

Offline demo. No API key needed. It uses a rule-based stand-in, **not a language model**, so the whole pipeline
(tools, loop, guardrails) can be tried and tested without network access:

```bash
python -m agent --trace
```

With a real model through the Anthropic API:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # your key
export ANTHROPIC_MODEL=...          # optional, a model available to your account
python -m agent --llm anthropic
```

Live mode needs a key and is not part of the automated tests.

## Tests

```bash
python -m unittest
```

42 tests, no network and no extra packages. They cover the retriever, the guardrails, the agent loop with a scripted
model (tool calls, repairs, step limits, invented citations) and the adapter against a fake client.

## Layout

| Path | Purpose |
| --- | --- |
| `agent/core.py` | The review loop and the system prompt. |
| `agent/tools.py` | Tool definitions and a safe dispatcher. |
| `agent/retrieval.py`, `agent/guidelines.py` | Guideline sections and BM25 search. |
| `agent/guardrails.py` | Validation of the final answer, PII masking. |
| `agent/anthropic_llm.py` | Adapter for the Anthropic Messages API with tool use. |
| `agent/offline_demo.py` | Rule-based stand-in used for the demo and tests. |
| `data/` | Example guidelines, annotator statistics and submissions (all fictional). |

## Limitations and next steps

- Retrieval is lexical (BM25) and English-oriented. Embeddings and a vector store are not implemented, the
  `search(query, k)` interface is small so they can be added.
- Annotator statistics come from a JSON file. Connecting to the Annotation Quality API over HTTP is a next step.
- There is no MCP server yet, the tools use plain function calling.
- The guardrails check format and grounding, not whether a verdict is correct. There is no evaluation set that
  measures the accuracy of a real model.

## Notes

Written with AI-assisted tooling. I chose the structure and the guardrails, then ran, reviewed and tested it myself.
The guidelines and data are fictional examples. MIT licensed.
