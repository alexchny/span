import shlex
import subprocess
from typing import Any, TypedDict

from span.models.tools import ToolResult
from span.tools.base import Tool


class ProgramRules(TypedDict):
    allowed_flags: set[str]
    allowed_positional: bool


ALLOWED_PROGRAMS: dict[str, ProgramRules] = {
    "pytest": {
        "allowed_flags": {"-v", "-x", "-q", "--version", "--tb=short", "--tb=long", "--lf", "--ff"},
        "allowed_positional": True,
    },
    "ruff": {
        "allowed_flags": {"check", "format", "--fix"},
        "allowed_positional": True,
    },
    "mypy": {
        "allowed_flags": {"--strict", "--no-error-summary"},
        "allowed_positional": True,
    },
    "python": {
        "allowed_flags": {"-m", "-c"},
        "allowed_positional": True,
    },
    "git": {
        "allowed_flags": {"status", "diff", "log", "show"},
        "allowed_positional": True,
    },
}


class RunShellTool(Tool):
    @property
    def name(self) -> str:
        return "run_shell"

    @property
    def description(self) -> str:
        return "Run restricted shell commands (pytest, ruff, mypy, python -m, git status/diff/log)"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "command": {
                "type": "string",
                "description": "Shell command to execute (must be in allowlist)",
                "required": True,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        command = kwargs["command"]

        try:
            args = shlex.split(command)
        except ValueError as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to parse command: {e}",
            )

        if not args:
            return ToolResult(
                success=False,
                output="",
                error="Empty command",
            )

        program = args[0]

        if program not in ALLOWED_PROGRAMS:
            return ToolResult(
                success=False,
                output="",
                error=f"Program not allowed: {program}. Allowed: {', '.join(ALLOWED_PROGRAMS.keys())}",
            )

        validation_error = self._validate_args(program, args[1:])
        if validation_error:
            return ToolResult(
                success=False,
                output="",
                error=validation_error,
            )

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300,
            )

            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr

            return ToolResult(
                success=result.returncode == 0,
                output=output.strip(),
                error=None if result.returncode == 0 else f"Command exited with code {result.returncode}",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error="Command timed out after 300 seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to execute command: {e}",
            )

    def _validate_args(self, program: str, args: list[str]) -> str | None:
        rules = ALLOWED_PROGRAMS[program]
        allowed_flags = rules["allowed_flags"]
        allowed_positional = rules["allowed_positional"]

        for arg in args:
            if arg.startswith("-") or arg in {"check", "format", "status", "diff", "log", "show"}:
                if arg not in allowed_flags:
                    return f"Flag not allowed for {program}: {arg}"
            else:
                if not allowed_positional:
                    return f"Positional arguments not allowed for {program}"

                if ".." in arg or arg.startswith("/"):
                    return f"Suspicious path in argument: {arg}"

        return None
