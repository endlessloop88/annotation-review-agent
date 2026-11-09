"""The two tools the agent can call, plus the dispatcher that runs them safely."""
import json
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import BM25Retriever

TOOL_DEFINITIONS = [
    {
        "name": "search_guidelines",
        "description": "Search the written labelling guidelines. Returns the most relevant sections with their ids.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up, in a few words."}},
            "required": ["query"],
        },
    },
    {
        "name": "get_annotator_stats",
        "description": "Get review statistics for an annotator: reviewed count, approval rate and correction rate.",
        "parameters": {
            "type": "object",
            "properties": {"annotator_id": {"type": "string"}},
            "required": ["annotator_id"],
        },
    },
]

SNIPPET_LENGTH = 700


@dataclass
class ToolContext:
    retriever: BM25Retriever
    stats: dict
    retrieved_ids: set[str] = field(default_factory=set)  # every section the model has actually seen


def load_stats(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_tool(name: str, arguments: dict, context: ToolContext) -> str:
    """Run one tool call and return a JSON string. Errors are returned as data, never raised to the loop."""
    try:
        if name == "search_guidelines":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return json.dumps({"error": "'query' must be a non-empty string"})
            results = []
            for section, score in context.retriever.search(query):
                context.retrieved_ids.add(section.id)
                results.append(
                    {"id": section.id, "title": section.title, "text": section.text[:SNIPPET_LENGTH], "score": round(score, 3)}
                )
            return json.dumps({"results": results})
        if name == "get_annotator_stats":
            annotator_id = arguments.get("annotator_id")
            record = context.stats.get(annotator_id) if isinstance(annotator_id, str) else None
            if record is None:
                return json.dumps({"error": f"unknown annotator: {annotator_id}"})
            return json.dumps(record)
        return json.dumps({"error": f"unknown tool: {name}"})
    except Exception as error:  # a broken tool must not crash the review
        return json.dumps({"error": f"tool failed: {type(error).__name__}"})
