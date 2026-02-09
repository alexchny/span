from pathlib import Path
from unittest.mock import MagicMock, patch

from span.config import Config
from span.context.repo_map import RepoMap
from span.core.agent import Agent
from span.core.verifier import VerificationResult, Verifier
from span.events.stream import EventStream
from span.llm.client import LLMClient


def test_run_with_plan_approval(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    plan_response = MagicMock()
    llm_client.extract_text.return_value = "Test plan"

    exec_response = MagicMock()
    llm_client.has_tool_use.return_value = False

    llm_client.send_message.side_effect = [plan_response, exec_response]

    with patch("builtins.input", return_value="y"):
        state = agent.run("Test task", show_plan=True)

    assert state.session_id
    assert state.original_task == "Test task"


def test_run_with_plan_rejection(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    plan_response = MagicMock()
    llm_client.extract_text.return_value = "Test plan"
    llm_client.send_message.return_value = plan_response

    with patch("builtins.input", return_value="n"):
        state = agent.run("Test task", show_plan=True)

    assert state.original_task == "Test task"
    assert len(state.changes) == 0


def test_execute_loop_with_tool_calls(tmp_path: Path) -> None:
    config = Config(max_steps=2)
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    plan_response = MagicMock()
    llm_client.extract_text.return_value = "Test plan"

    tool_response = MagicMock()
    llm_client.has_tool_use.side_effect = [True, False]
    llm_client.extract_tool_calls.return_value = [
        {
            "id": "call_1",
            "name": "read_file",
            "input": {"path": "test.py"}
        }
    ]

    llm_client.send_message.side_effect = [plan_response, tool_response, MagicMock()]

    test_file = tmp_path / "test.py"
    test_file.write_text("test content")

    with patch.object(agent.read_file_tool, "execute") as mock_read:
        mock_result = MagicMock()
        mock_result.to_content.return_value = [{"type": "text", "text": "file contents"}]
        mock_read.return_value = mock_result

        state = agent.run("Read a file", show_plan=False)

    assert state.tool_call_count == 1


def test_execute_loop_stops_at_turn_limit(tmp_path: Path) -> None:
    config = Config(max_steps=2)
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    plan_response = MagicMock()
    llm_client.extract_text.return_value = "Test plan"

    tool_response = MagicMock()
    llm_client.has_tool_use.return_value = True
    llm_client.extract_tool_calls.return_value = []

    llm_client.send_message.side_effect = [plan_response, tool_response, tool_response, tool_response]

    state = agent.run("Infinite task", show_plan=False)

    assert state.turn_count == 2


def test_show_diff_with_changes(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import ChangeOp
    changes = [
        ChangeOp("file1.py", "+new1", "-new1", 1.0, 1),
        ChangeOp("file2.py", "+new2", "-new2", 2.0, 2),
    ]

    with patch("builtins.print") as mock_print:
        agent._show_diff(changes)

    assert mock_print.call_count > 0


def test_handle_revision(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState, ChangeOp
    state = AgentState(
        session_id="test",
        messages=[],
        original_task="Original task",
        tool_call_count=5
    )
    state.changes = [ChangeOp("test.py", "+fix", "-fix", 1.0, 1)]
    state.last_errors = ["Error 1"]

    plan_response = MagicMock()
    llm_client.extract_text.return_value = "Revised plan"

    final_response = MagicMock()
    llm_client.has_tool_use.return_value = False

    llm_client.send_message.side_effect = [plan_response, final_response]

    new_state = agent.handle_revision(state, "Fix it differently", show_plan=False)

    assert new_state.session_id
    assert "Original task" in agent._build_run_summary(state)
    assert "Fix it differently" not in agent._build_run_summary(state)


def test_execute_tool_unknown_tool() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState
    state = AgentState(session_id="test", messages=[])

    tool_call = {"name": "unknown_tool", "input": {}}
    result = agent._execute_tool(tool_call, state)

    assert isinstance(result, list)
    assert "Unknown tool" in result[0]["text"]


def test_execute_patch_apply_failure(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState
    state = AgentState(session_id="test", messages=[])

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(
            success=False,
            error="Patch failed to apply",
            reverse_diff=None
        )

        tool_input = {"path": "test.py", "diff": "+ invalid"}
        result = agent._execute_patch_with_verification(tool_input, state)

        assert isinstance(result, list)
        assert "Error" in result[0]["text"]
        assert len(state.changes) == 0


def test_execute_patch_no_reverse_diff_existing_file(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState
    state = AgentState(session_id="test", messages=[])

    existing_file = tmp_path / "test.py"
    existing_file.write_text("old content")

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(
            success=True,
            error=None,
            reverse_diff=None
        )

        with patch.object(agent.verifier, "verify_patch") as mock_verify:
            mock_verify.return_value = VerificationResult(passed=True, errors=[])

            tool_input = {"path": str(existing_file), "diff": "+ new"}
            result = agent._execute_patch_with_verification(tool_input, state)

            assert isinstance(result, list)
            assert "Failed to generate reverse diff" in result[0]["text"]
            assert len(state.changes) == 0


def test_execute_patch_new_file_no_reverse_diff(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState
    state = AgentState(session_id="test", messages=[])

    new_file = tmp_path / "new_file.py"

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(
            success=True,
            error=None,
            reverse_diff=None
        )

        with patch.object(agent.verifier, "verify_patch") as mock_verify:
            mock_verify.return_value = VerificationResult(passed=True, errors=[])

            with patch.object(agent.verifier, "check_lint") as mock_lint:
                mock_lint.return_value = VerificationResult(passed=True, errors=[])

                tool_input = {"path": str(new_file), "diff": "@@ -0,0 +1 @@\n+print('hello')"}
                result = agent._execute_patch_with_verification(tool_input, state)

                assert isinstance(result, list)
                assert "SUCCESS:" in result[0]["text"]
                assert len(state.changes) == 1


def test_finalize_with_final_check_warnings(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    from span.core.agent import AgentState, ChangeOp
    state = AgentState(session_id="test", messages=[])
    state.changes = [ChangeOp("test.py", "+new", "-new", 1.0, 1)]

    verifier.verify_final.return_value = VerificationResult(
        passed=False,
        errors=["Type error in test.py"]
    )

    with patch("builtins.input", return_value="y"):
        with patch("builtins.print") as mock_print:
            result = agent.finalize(state)

            assert result is True
            printed = "".join(str(call) for call in mock_print.call_args_list)
            assert "Type error" in printed or "Final checks" in printed
