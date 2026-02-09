from pathlib import Path
from unittest.mock import MagicMock, patch

from span.config import Config
from span.context.repo_map import RepoMap
from span.core.agent import Agent, AgentState, ChangeOp, RevertError
from span.core.verifier import VerificationResult, Verifier
from span.events.stream import EventStream
from span.llm.client import LLMClient


def test_change_op_creation() -> None:
    op = ChangeOp(
        path="test.py",
        forward_diff="+ new line",
        reverse_diff="- new line",
        timestamp=123.45,
        step_id=1,
    )

    assert op.path == "test.py"
    assert op.step_id == 1


def test_agent_state_initialization() -> None:
    state = AgentState(
        session_id="test123",
        messages=[],
        original_task="Fix bug",
    )

    assert state.session_id == "test123"
    assert state.turn_count == 0
    assert state.tool_call_count == 0
    assert state.patch_attempt_count == 0
    assert len(state.changes) == 0


def test_agent_initialization() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    assert agent.config == config
    assert agent.repo_map == repo_map


def test_check_limits_max_turns() -> None:
    config = Config(max_steps=5)
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[], turn_count=5)

    limit = agent._check_limits(state)

    assert limit == "max_turns"


def test_check_limits_max_tool_calls() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[], tool_call_count=50)

    limit = agent._check_limits(state)

    assert limit == "max_tool_calls"


def test_check_limits_no_limit() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    limit = agent._check_limits(state)

    assert limit is None


def test_execute_tool_read_file() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    with patch.object(agent.read_file_tool, "execute") as mock_execute:
        mock_result = MagicMock(success=True, output="file contents")
        mock_result.to_content.return_value = [{"type": "text", "text": "file contents"}]
        mock_execute.return_value = mock_result

        tool_call = {"name": "read_file", "input": {"path": "test.py"}}
        result = agent._execute_tool(tool_call, state)

        assert isinstance(result, list)
        assert result == [{"type": "text", "text": "file contents"}]
        mock_execute.assert_called_once_with(path="test.py")


def test_execute_patch_with_verification_success(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    test_file = tmp_path / "test.py"
    test_file.write_text("old content")

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(
            success=True, reverse_diff="- new\n+ old", error=None
        )

        with patch.object(agent.verifier, "verify_patch") as mock_verify:
            mock_verify.return_value = VerificationResult(passed=True, errors=[])

            tool_input = {"path": str(test_file), "diff": "+ new line"}
            result = agent._execute_patch_with_verification(tool_input, state)

            assert isinstance(result, list)
            assert len(result) > 0
            assert "applied" in result[0]["text"].lower()
            assert len(state.changes) == 1
            assert state.changes[0].path == str(test_file)


def test_execute_patch_with_verification_failure(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    test_file = tmp_path / "test.py"
    test_file.write_text("old content")

    call_count = [0]

    def mock_execute_side_effect(path: str, diff: str) -> MagicMock:
        call_count[0] += 1
        if call_count[0] == 1:
            return MagicMock(success=True, reverse_diff="- new\n+ old", error=None, original_content="old content")
        else:
            return MagicMock(success=True, reverse_diff=None, error=None, original_content=None)

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.side_effect = mock_execute_side_effect

        with patch.object(agent.verifier, "verify_patch") as mock_verify:
            mock_verify.return_value = VerificationResult(
                passed=False, errors=["Syntax error"]
            )

            tool_input = {"path": str(test_file), "diff": "+ new line"}
            result = agent._execute_patch_with_verification(tool_input, state)

            assert isinstance(result, list)
            assert len(result) > 0
            assert "reverted" in result[0]["text"].lower()
            assert "syntax error" in result[0]["text"].lower()
            assert len(state.changes) == 0
            assert mock_apply.call_count == 1


def test_revert_all_changes() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    changes = [
        ChangeOp("file1.py", "+new1", "-new1", 1.0, 1),
        ChangeOp("file2.py", "+new2", "-new2", 2.0, 2),
        ChangeOp("file3.py", "+new3", "-new3", 3.0, 3),
    ]

    with patch.object(agent, "_apply_reverse_diff") as mock_revert:
        agent.revert_all(changes)

        assert mock_revert.call_count == 3
        mock_revert.assert_any_call("file3.py", "-new3")
        mock_revert.assert_any_call("file2.py", "-new2")
        mock_revert.assert_any_call("file1.py", "-new1")


def test_build_run_summary() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    state = AgentState(
        session_id="test",
        messages=[],
        original_task="Fix login bug",
        tool_call_count=5,
    )

    state.changes = [
        ChangeOp("auth.py", "+fix", "-fix", 1.0, 1),
        ChangeOp("login.py", "+fix2", "-fix2", 2.0, 2),
    ]

    state.last_errors = ["Syntax error in line 10"]

    summary = agent._build_run_summary(state)

    assert "Fix login bug" in summary
    assert "Steps taken: 5" in summary
    assert "auth.py" in summary
    assert "login.py" in summary
    assert "Syntax error" in summary


def test_get_plan() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    mock_response = MagicMock()
    llm_client.send_message.return_value = mock_response
    llm_client.extract_text.return_value = "1. Read file\n2. Fix bug\n3. Test"

    plan = agent._get_plan("Fix the bug", "session123")

    assert "Read file" in plan
    assert "Fix bug" in plan
    llm_client.send_message.assert_called_once()
    event_stream.append.assert_called_once()


def test_finalize_no_changes() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    result = agent.finalize(state)

    assert result is False


def test_finalize_with_changes_keep(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    state.changes = [ChangeOp("test.py", "+new", "-new", 1.0, 1)]

    verifier.verify_final.return_value = VerificationResult(passed=True, errors=[])

    with patch("builtins.input", return_value="y"):
        result = agent.finalize(state)

        assert result is True
        assert len(state.changes) == 0


def test_finalize_with_changes_revert(tmp_path: Path) -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)
    state = AgentState(session_id="test", messages=[])

    state.changes = [ChangeOp("test.py", "+new", "-new", 1.0, 1)]

    verifier.verify_final.return_value = VerificationResult(passed=True, errors=[])

    with patch("builtins.input", return_value="n"):
        with patch.object(agent, "revert_all") as mock_revert:
            result = agent.finalize(state)

            assert result is False
            mock_revert.assert_called_once_with(state.changes)


def test_revert_error_exception() -> None:
    err = RevertError([
        ("file1.py", "diff1", "Failed to revert file1.py"),
        ("file2.py", "diff2", "Failed to revert file2.py"),
    ])
    assert "file1.py" in str(err)
    assert "file2.py" in str(err)
    assert len(err.failed_ops) == 2


def test_revert_all_raises_on_failure() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    changes = [
        ChangeOp("file1.py", "f1", "r1", 1.0, 1),
        ChangeOp("file2.py", "f2", "r2", 2.0, 2),
    ]

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(success=False, error="Revert failed")

        import pytest
        with pytest.raises(RevertError) as exc_info:
            agent.revert_all(changes)

        assert len(exc_info.value.failed_ops) == 2
        assert "file1.py" in str(exc_info.value)


def test_apply_reverse_diff_returns_success() -> None:
    config = Config()
    repo_map = MagicMock(spec=RepoMap)
    llm_client = MagicMock(spec=LLMClient)
    verifier = MagicMock(spec=Verifier)
    event_stream = MagicMock(spec=EventStream)

    agent = Agent(config, repo_map, llm_client, verifier, event_stream)

    with patch.object(agent.apply_patch_tool, "execute") as mock_apply:
        mock_apply.return_value = MagicMock(success=True)
        result = agent._apply_reverse_diff("test.py", "reverse diff")
        assert result is True

        mock_apply.return_value = MagicMock(success=False)
        result = agent._apply_reverse_diff("test.py", "reverse diff")
        assert result is False
