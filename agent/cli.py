"""Command line entry point: python -m agent [submissions.json] [--llm demo|anthropic] [--trace]"""
import argparse
import json
import sys
from pathlib import Path

from .core import ReviewFailed, review
from .guidelines import load_guidelines
from .offline_demo import OfflineDemoLLM
from .retrieval import BM25Retriever
from .tools import ToolContext, load_stats

ROOT = Path(__file__).resolve().parent.parent


def build_llm(kind: str):
    if kind == "anthropic":
        from .anthropic_llm import AnthropicLLM

        return AnthropicLLM()
    return OfflineDemoLLM()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review labelling submissions against written guidelines.")
    parser.add_argument("submissions", nargs="?", default=str(ROOT / "data" / "submissions.json"))
    parser.add_argument("--guidelines", default=str(ROOT / "data" / "guidelines"))
    parser.add_argument("--stats", default=str(ROOT / "data" / "annotator_stats.json"))
    parser.add_argument("--llm", choices=["demo", "anthropic"], default="demo", help="demo works offline; anthropic needs an API key")
    parser.add_argument("--trace", action="store_true", help="include the tool calls in the output")
    args = parser.parse_args(argv)

    submissions = json.loads(Path(args.submissions).read_text(encoding="utf-8"))
    if isinstance(submissions, dict):
        submissions = [submissions]
    context = ToolContext(retriever=BM25Retriever(load_guidelines(args.guidelines)), stats=load_stats(args.stats))
    llm = build_llm(args.llm)

    failures = 0
    for submission in submissions:
        entry: dict = {"id": submission.get("id")}
        try:
            verdict, trace = review(submission, llm, context)
            entry["verdict"] = verdict.to_dict()
            if args.trace:
                entry["tool_calls"] = trace
        except (ReviewFailed, ValueError) as error:
            failures += 1
            entry["error"] = str(error)
        print(json.dumps(entry, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
