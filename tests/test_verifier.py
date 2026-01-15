from pathlib import Path
from unittest.mock import MagicMock, patch

from span.context.repo_map import RepoMap
from span.core.verifier import VerificationResult, Verifier


def test_verification_result_passed() -> None:
    result = VerificationResult(passed=True, errors=[])
    assert result.passed
    assert result.errors == []


def test_verification_result_failed() -> None:
    result = VerificationResult(passed=False, errors=["error 1"])
    assert not result.passed
    assert result.errors == ["error 1"]


def test_check_syntax_valid(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "valid.py"
    test_file.write_text("def foo():\n    return 42\n")

    result = verifier.check_syntax(str(test_file))

    assert result.passed
    assert result.errors == []


def test_check_syntax_invalid(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "invalid.py"
    test_file.write_text("def foo(\n")

    result = verifier.check_syntax(str(test_file))

    assert not result.passed
    assert len(result.errors) == 1
    assert "Syntax error" in result.errors[0]


def test_check_syntax_file_not_found(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    result = verifier.check_syntax(str(tmp_path / "nonexistent.py"))

    assert not result.passed
    assert "File not found" in result.errors[0]


def test_check_lint_success(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "clean.py"
    test_file.write_text("def foo() -> int:\n    return 42\n")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.check_lint([str(test_file)])

    assert result.passed
    assert result.errors == []


def test_check_lint_failure(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "messy.py"
    test_file.write_text("import os\n")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="F401 'os' imported but unused",
            stderr=""
        )
        result = verifier.check_lint([str(test_file)])

    assert not result.passed
    assert "Lint errors" in result.errors[0]


def test_check_tests_success(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    repo_map.find_affected_tests.return_value = ["tests/test_foo.py"]
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.check_tests(["src/foo.py"], full=False)

    assert result.passed
    assert result.errors == []


def test_check_tests_failure(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    repo_map.find_affected_tests.return_value = ["tests/test_foo.py"]
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar",
            stderr=""
        )
        result = verifier.check_tests(["src/foo.py"], full=False)

    assert not result.passed
    assert "Test failures" in result.errors[0]


def test_check_tests_no_affected_uses_fallback(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    repo_map.find_affected_tests.return_value = []
    verifier = Verifier(repo_map, ["tests/"], ["tests/test_core.py"])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.check_tests(["src/foo.py"], full=False)

    assert result.passed
    mock_run.assert_called_once()
    assert "tests/test_core.py" in mock_run.call_args[0][0]


def test_check_tests_full_mode(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.check_tests(["src/foo.py"], full=True)

    assert result.passed
    mock_run.assert_called_once()
    assert mock_run.call_args[0][0] == ["pytest", "-q"]


def test_check_types_success(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.check_types()

    assert result.passed
    assert result.errors == []


def test_check_types_failure(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="error: Incompatible types",
            stderr=""
        )
        result = verifier.check_types()

    assert not result.passed
    assert "Type errors" in result.errors[0]


def test_verify_patch_all_pass(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    repo_map.find_affected_tests.return_value = []
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "good.py"
    test_file.write_text("def foo() -> int:\n    return 42\n")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.verify_patch(str(test_file))

    assert result.passed


def test_verify_patch_syntax_fails(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    test_file = tmp_path / "bad.py"
    test_file.write_text("def foo(\n")

    result = verifier.verify_patch(str(test_file))

    assert not result.passed
    assert "Syntax error" in result.errors[0]


def test_verify_final_success(tmp_path: Path) -> None:
    repo_map = MagicMock(spec=RepoMap)
    verifier = Verifier(repo_map, ["tests/"], [])

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = verifier.verify_final()

    assert result.passed
