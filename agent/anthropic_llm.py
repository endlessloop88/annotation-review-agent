"""Adapter for the Anthropic Messages API with tool use.

The SDK is imported lazily, so the rest of the project (and the tests) work without it installed.
Set ANTHROPIC_API_KEY, and optionally ANTHROPIC_MODEL to a model available to your account.
"""
import os

from .types import LLMResponse, ToolCall

DEFAULT_MODEL = "claude-haiku-5-5"


class AnthropicLLM:
    def __init__(self, model: str | None = None, client=None, max_tokens: int = 1024):
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        if client is None:
            import anthropic  # needs `pip install anthropic`; reads ANTHROPIC_API_KEY from the environment

            client = anthropic.Anthropic()
        self.client = client

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[
                {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
                for tool in tools
            ],
            messages=self.to_provider_messages(messages),
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        return LLMResponse(text=text, tool_calls=calls)

    @staticmethod
    def to_provider_messages(messages: list[dict]) -> list[dict]:
        """Translate the neutral message format. Tool results travel inside a user message."""
        converted: list[dict] = []
        for message in messages:
            role = message["role"]
            if role == "user":
                converted.append({"role": "user", "content": message["content"]})
            elif role == "assistant":
                blocks = []
                if message.get("text"):
                    blocks.append({"type": "text", "text": message["text"]})
                for call in message.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]})
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": message["content"]}
                previous = converted[-1] if converted else None
                if (
                    previous
                    and previous["role"] == "user"
                    and isinstance(previous["content"], list)
                    and all(item.get("type") == "tool_result" for item in previous["content"])
                ):
                    previous["content"].append(block)  # several tool calls in one turn share one user message
                else:
                    converted.append({"role": "user", "content": [block]})
        return converted
