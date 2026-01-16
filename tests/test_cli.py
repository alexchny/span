import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from span.cli import cli, diff, logs, status
from span.config import Config
from span.core.agent import AgentState, ChangeOp
from span.models.events import Event


def test_cli_no_command() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code in (0, 2)
    assert "Usage:" in result.output


def test_run_missing_api_key(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=True):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                result = runner.invoke(cli, ["run", "Fix the bug"])
                assert result.exit_code == 1
                assert "ANTHROPIC_API_KEY not found" in result.output


def test_run_with_opus_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent.return_value = mock_agent_instance

                                    runner.invoke(cli, ["run", "--opus", "Fix the bug"])

                                    assert mock_config.model == "claude-3-opus-20240229"


def test_run_with_full_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent.return_value = mock_agent_instance

                                    runner.invoke(cli, ["run", "--full", "Fix the bug"])

                                    assert mock_config.verification.pytest is True


def test_run_with_verbose_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent.return_value = mock_agent_instance

                                    old_verbose = os.environ.pop("SPAN_VERBOSE", None)
                                    try:
                                        runner.invoke(cli, ["run", "--verbose", "Fix the bug"])
                                        assert os.environ.get("SPAN_VERBOSE") == "1"
                                    finally:
                                        if old_verbose is not None:
                                            os.environ["SPAN_VERBOSE"] = old_verbose
                                        else:
                                            os.environ.pop("SPAN_VERBOSE", None)


def test_run_with_plan_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent.return_value = mock_agent_instance

                                    runner.invoke(cli, ["run", "--plan", "Fix the bug"])

                                    mock_agent_instance.run.assert_called_once()
                                    call_kwargs = mock_agent_instance.run.call_args[1]
                                    assert call_kwargs.get("show_plan") is True


def test_run_successful_with_changes(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_state.changes = [
                                        ChangeOp("test.py", "+new", "-new", 1.0, 1)
                                    ]

                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent_instance.finalize.return_value = True
                                    mock_agent.return_value = mock_agent_instance

                                    result = runner.invoke(cli, ["run", "Fix the bug"])

                                    assert result.exit_code == 0
                                    mock_agent_instance.run.assert_called_once()
                                    mock_agent_instance.finalize.assert_called_once()


def test_run_with_revision(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_state = AgentState(
                                        session_id="test",
                                        messages=[],
                                        original_task="Fix bug",
                                    )
                                    mock_state.changes = [
                                        ChangeOp("test.py", "+new", "-new", 1.0, 1)
                                    ]

                                    mock_new_state = AgentState(
                                        session_id="test2",
                                        messages=[],
                                        original_task="Fix differently",
                                    )

                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.return_value = mock_state
                                    mock_agent_instance.finalize.side_effect = [
                                        False,
                                        True,
                                    ]
                                    mock_agent_instance.handle_revision.return_value = (
                                        mock_new_state
                                    )
                                    mock_agent.return_value = mock_agent_instance

                                    runner.invoke(
                                        cli, ["run", "Fix the bug"], input="Fix differently\n"
                                    )

                                    assert mock_agent_instance.handle_revision.call_count == 1


def test_run_keyboard_interrupt(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("span.cli.load_config") as mock_load:
                mock_config = Config()
                mock_load.return_value = mock_config

                with patch("span.cli.RepoMap"):
                    with patch("span.cli.LLMClient"):
                        with patch("span.cli.Verifier"):
                            with patch("span.cli.EventStream"):
                                with patch("span.cli.Agent") as mock_agent:
                                    mock_agent_instance = MagicMock()
                                    mock_agent_instance.run.side_effect = KeyboardInterrupt()
                                    mock_agent.return_value = mock_agent_instance

                                    result = runner.invoke(cli, ["run", "Fix the bug"])

                                    assert result.exit_code == 1
                                    assert "Interrupted" in result.output


def test_status_no_events(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = []
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(status)

            assert result.exit_code == 0
            assert "No sessions found" in result.output


def test_status_with_events(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create(
                    "plan", session_id="test123", task="Fix bug", plan="1. Fix it"
                ),
                Event.create(
                    "tool_result",
                    session_id="test123",
                    result=[{"type": "text", "text": "Patch applied and verified"}],
                ),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(status)

            assert result.exit_code == 0
            assert "test123" in result.output
            assert "Fix bug" in result.output
            assert "Changes: 1" in result.output


def test_logs_no_events(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = []
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(logs)

            assert result.exit_code == 0
            assert "No events found" in result.output


def test_logs_with_events(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create("session_start", session_id="test123", task="Fix bug"),
                Event.create("plan", session_id="test123", plan="Test plan"),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(logs)

            assert result.exit_code == 0
            assert "session_start" in result.output
            assert "plan" in result.output


def test_logs_with_session_filter(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create("session_start", session_id="test123", task="Fix bug"),
                Event.create("session_start", session_id="test456", task="Other task"),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(logs, ["--session", "test123"])

            assert result.exit_code == 0
            assert "test123" in result.output
            assert "test456" not in result.output


def test_logs_with_tail(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create("event1", session_id="test"),
                Event.create("event2", session_id="test"),
                Event.create("event3", session_id="test"),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(logs, ["--tail", "2"])

            assert result.exit_code == 0
            assert "event2" in result.output
            assert "event3" in result.output


def test_diff_no_events(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = []
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(diff)

            assert result.exit_code == 0
            assert "No events found" in result.output


def test_diff_no_changes(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create("session_start", session_id="test123", task="Fix bug"),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(diff)

            assert result.exit_code == 0
            assert "No changes found" in result.output


def test_diff_with_changes(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create(
                    "tool_call",
                    session_id="test123",
                    tool="apply_patch",
                    args={"path": "test.py", "diff": "+ new line"},
                ),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(diff)

            assert result.exit_code == 0
            assert "test.py" in result.output
            assert "+ new line" in result.output


def test_diff_with_session_filter(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        with patch("span.cli.EventStream") as mock_stream:
            events = [
                Event.create(
                    "tool_call",
                    session_id="test123",
                    tool="apply_patch",
                    args={"path": "test.py", "diff": "+ new line"},
                ),
                Event.create(
                    "tool_call",
                    session_id="test456",
                    tool="apply_patch",
                    args={"path": "other.py", "diff": "+ other line"},
                ),
            ]

            mock_stream_instance = MagicMock()
            mock_stream_instance.read_all.return_value = events
            mock_stream.return_value = mock_stream_instance

            result = runner.invoke(diff, ["--session", "test123"])

            assert result.exit_code == 0
            assert "test.py" in result.output
            assert "other.py" not in result.output
