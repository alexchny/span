from unittest.mock import MagicMock, patch

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from span.llm.client import LLMClient
from span.llm.prompts import EXECUTE_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT


def test_plan_system_prompt() -> None:
    assert "Span" in PLAN_SYSTEM_PROMPT
    assert "plan" in PLAN_SYSTEM_PROMPT.lower()


def test_execute_system_prompt() -> None:
    assert "Span" in EXECUTE_SYSTEM_PROMPT
    assert "apply_patch" in EXECUTE_SYSTEM_PROMPT
    assert "verification" in EXECUTE_SYSTEM_PROMPT.lower()


@patch("span.llm.client.Anthropic")
def test_llm_client_init(mock_anthropic: MagicMock) -> None:
    client = LLMClient(model="claude-sonnet-4-20250514", api_key="test-key")

    assert client.model == "claude-sonnet-4-20250514"
    mock_anthropic.assert_called_once_with(api_key="test-key")


@patch("span.llm.client.Anthropic")
def test_send_message(mock_anthropic: MagicMock) -> None:
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    mock_response = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello")],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )
    mock_client.messages.create.return_value = mock_response

    client = LLMClient(model="claude-sonnet-4-20250514", api_key="test-key")
    messages = [{"role": "user", "content": "Hi"}]
    response = client.send_message(messages, system="You are helpful")

    assert response.id == "msg_123"
    mock_client.messages.create.assert_called_once()


@patch("span.llm.client.Anthropic")
def test_extract_text(mock_anthropic: MagicMock) -> None:
    client = LLMClient(model="claude-sonnet-4-20250514", api_key="test-key")

    message = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            TextBlock(type="text", text="Hello "),
            TextBlock(type="text", text="world"),
        ],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    text = client.extract_text(message)
    assert text == "Hello world"


@patch("span.llm.client.Anthropic")
def test_extract_tool_calls(mock_anthropic: MagicMock) -> None:
    client = LLMClient(model="claude-sonnet-4-20250514", api_key="test-key")

    message = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            TextBlock(type="text", text="Let me read that file"),
            ToolUseBlock(
                type="tool_use",
                id="tool_1",
                name="read_file",
                input={"path": "test.py"},
            ),
        ],
        model="claude-sonnet-4-20250514",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    tool_calls = client.extract_tool_calls(message)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "read_file"
    assert tool_calls[0]["input"]["path"] == "test.py"


@patch("span.llm.client.Anthropic")
def test_has_tool_use(mock_anthropic: MagicMock) -> None:
    client = LLMClient(model="claude-sonnet-4-20250514", api_key="test-key")

    message_with_tool = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[
            ToolUseBlock(
                type="tool_use",
                id="tool_1",
                name="read_file",
                input={"path": "test.py"},
            ),
        ],
        model="claude-sonnet-4-20250514",
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    message_without_tool = Message(
        id="msg_124",
        type="message",
        role="assistant",
        content=[TextBlock(type="text", text="Hello")],
        model="claude-sonnet-4-20250514",
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
    )

    assert client.has_tool_use(message_with_tool) is True
    assert client.has_tool_use(message_without_tool) is False
