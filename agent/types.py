"""Provider-neutral data types shared by the agent and the LLM adapters."""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    """Anything that can answer a conversation, optionally by calling tools.

    Messages are plain dicts:
      {"role": "user", "content": str}
      {"role": "assistant", "text": str, "tool_calls": [{"id", "name", "arguments"}]}
      {"role": "tool", "tool_call_id": str, "name": str, "content": str}
    """

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...
