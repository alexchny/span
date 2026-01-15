import os
from collections.abc import Iterator
from typing import Any

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageStreamEvent,
    TextBlock,
    ToolUseBlock,
)


class LLMClient:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def send_message(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> Message:
        return self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools or [],
        )

    def stream_message(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 8192,
    ) -> Iterator[MessageStreamEvent]:
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools or [],
        ) as stream:
            yield from stream

    def extract_text(self, message: Message) -> str:
        text_parts = []
        for block in message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        return "".join(text_parts)

    def extract_tool_calls(self, message: Message) -> list[dict[str, Any]]:
        tool_calls = []
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
        return tool_calls

    def has_tool_use(self, message: Message) -> bool:
        return any(isinstance(block, ToolUseBlock) for block in message.content)
