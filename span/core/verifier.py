import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from span.context.repo_map import RepoMap


@dataclass
class VerificationResult:
    passed: bool
    errors: list[str]


class Verifier:
    def __init__(
        self,
        repo_map: RepoMap,
        test_patterns: list[str],
        fallback_tests: list[str],
    ):
        self.repo_map = repo_map
        self.test_patterns = test_patterns
        self.fallback_tests = fallback_tests

    def check_syntax(self, file_path: str) -> VerificationResult:
        try:
            content = Path(file_path).read_text()
            ast.parse(content)
            return VerificationResult(passed=True, errors=[])
        except SyntaxError as e:
            return VerificationResult(
                passed=False,
                errors=[f"Syntax error in {file_path}:{e.lineno}: {e.msg}"]
            )
        except FileNotFoundError:
            return VerificationResult(
                passed=False,
                errors=[f"File not found: {file_path}"]
            )

    def check_lint(self, file_paths: list[str]) -> VerificationResult:
        try:
            result = subprocess.run(
                ["ruff", "check"] + file_paths,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return VerificationResult(passed=True, errors=[])
            else:
                return VerificationResult(
                    passed=False,
                    errors=[f"Lint errors:\n{result.stdout}"]
                )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                errors=["Linting timed out after 30 seconds"]
            )
        except FileNotFoundError:
            return VerificationResult(
                passed=False,
                errors=["ruff not found in PATH"]
            )

    def check_tests(self, modified_files: list[str], full: bool = False) -> VerificationResult:
        if full:
            test_files = []
        else:
            test_files = self.repo_map.find_affected_tests(
                modified_files=modified_files,
                test_patterns=self.test_patterns,
            )

            if not test_files and self.fallback_tests:
                test_files = self.fallback_tests

        if not test_files and not full:
            return VerificationResult(passed=True, errors=[])

        try:
            if full:
                cmd = ["pytest", "-q"]
            else:
                cmd = ["pytest", "-q"] + test_files

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                return VerificationResult(passed=True, errors=[])
            else:
                return VerificationResult(
                    passed=False,
                    errors=[f"Test failures:\n{result.stdout}\n{result.stderr}"]
                )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                errors=["Tests timed out after 120 seconds"]
            )
        except FileNotFoundError:
            return VerificationResult(
                passed=False,
                errors=["pytest not found in PATH"]
            )

    def check_types(self) -> VerificationResult:
        try:
            result = subprocess.run(
                ["mypy", ".", "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return VerificationResult(passed=True, errors=[])
            else:
                return VerificationResult(
                    passed=False,
                    errors=[f"Type errors:\n{result.stdout}"]
                )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                errors=["Type checking timed out after 60 seconds"]
            )
        except FileNotFoundError:
            return VerificationResult(
                passed=False,
                errors=["mypy not found in PATH"]
            )

    def verify_patch(self, modified_file: str) -> VerificationResult:
        syntax_result = self.check_syntax(modified_file)
        if not syntax_result.passed:
            return syntax_result

        lint_result = self.check_lint([modified_file])
        if not lint_result.passed:
            return lint_result

        test_result = self.check_tests([modified_file], full=False)
        if not test_result.passed:
            return test_result

        return VerificationResult(passed=True, errors=[])

    def verify_final(self) -> VerificationResult:
        types_result = self.check_types()
        if not types_result.passed:
            return types_result

        return VerificationResult(passed=True, errors=[])
