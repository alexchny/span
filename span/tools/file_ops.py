import re
import subprocess
from pathlib import Path
from typing import Any

from span.models.tools import ApplyPatchResult, ToolResult
from span.tools.base import Tool


class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file with line numbers"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "path": {
                "type": "string",
                "description": "Path to the file to read",
                "required": True,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        path = Path(kwargs["path"])

        if not path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File not found: {path}",
            )

        if not path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Path is not a file: {path}",
            )

        try:
            content = path.read_text()
            lines = content.split("\n")
            numbered = "\n".join(f"{i+1:6}|{line}" for i, line in enumerate(lines))
            return ToolResult(success=True, output=numbered)
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to read file: {e}",
            )


class ApplyPatchTool(Tool):
    LAZY_PATTERNS = [
        r"\.\.\..*rest of",
        r"\.\.\..*existing",
        r"\.\.\..*unchanged",
        r"#.*TODO",
        r"//.*TODO",
        r"pass\s*#.*placeholder",
    ]

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return "Apply a unified diff patch to a file. Must include ≥3 context lines before OR after changes."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "patch": {
                "type": "string",
                "description": "Unified diff format patch with proper headers and context",
                "required": True,
            },
        }

    def execute(self, **kwargs: Any) -> ApplyPatchResult:
        patch = kwargs["patch"]

        file_path = self._extract_file_path(patch)
        if not file_path:
            return ApplyPatchResult(
                success=False,
                output="",
                error="Could not extract file path from patch headers",
            )

        if not self._validate_patch(patch):
            return ApplyPatchResult(
                success=False,
                output="",
                error="Patch validation failed: insufficient context lines or lazy patterns detected",
            )

        reverse_diff = self._generate_reverse_diff(file_path, patch)

        result = subprocess.run(
            ["patch", "-p0"],
            input=patch.encode(),
            capture_output=True,
        )

        if result.returncode == 0:
            return ApplyPatchResult(
                success=True,
                output=f"Patch applied successfully to {file_path}",
                file_path=str(file_path),
                reverse_diff=reverse_diff,
            )
        else:
            error_output = result.stderr.decode() if result.stderr else "Unknown error"
            return ApplyPatchResult(
                success=False,
                output=error_output,
                error="Patch failed to apply",
            )

    def _extract_file_path(self, patch: str) -> Path | None:
        for line in patch.split("\n"):
            if line.startswith("---"):
                parts = line.split()
                if len(parts) >= 2:
                    path_str = parts[1].removeprefix("a/")
                    return Path(path_str)
        return None

    def _validate_patch(self, patch: str) -> bool:
        for pattern in self.LAZY_PATTERNS:
            if re.search(pattern, patch, re.IGNORECASE):
                return False

        hunks = self._extract_hunks(patch)
        for hunk in hunks:
            if not self._has_sufficient_context(hunk):
                return False

        return True

    def _extract_hunks(self, patch: str) -> list[str]:
        hunks: list[str] = []
        current_hunk: list[str] = []
        in_hunk = False

        for line in patch.split("\n"):
            if line.startswith("@@"):
                if current_hunk:
                    hunks.append("\n".join(current_hunk))
                current_hunk = [line]
                in_hunk = True
            elif in_hunk:
                if line.startswith("---") or line.startswith("+++") or line.startswith("diff"):
                    if current_hunk:
                        hunks.append("\n".join(current_hunk))
                    current_hunk = []
                    in_hunk = False
                else:
                    current_hunk.append(line)

        if current_hunk:
            hunks.append("\n".join(current_hunk))

        return hunks

    def _has_sufficient_context(self, hunk: str) -> bool:
        lines = hunk.split("\n")[1:]
        context_before = 0
        context_after = 0
        seen_change = False

        for line in lines:
            if line.startswith(" "):
                if not seen_change:
                    context_before += 1
                else:
                    context_after += 1
            elif line.startswith(("+", "-")):
                seen_change = True
                context_after = 0

        return context_before >= 3 or context_after >= 3

    def _generate_reverse_diff(self, file_path: Path, patch: str) -> str | None:
        if not file_path.exists():
            return None

        try:
            lines = []
            lines.append(f"--- {file_path}")
            lines.append(f"+++ {file_path}")

            for line in patch.split("\n"):
                if line.startswith("@@"):
                    lines.append(line)
                elif line.startswith("+"):
                    lines.append("-" + line[1:])
                elif line.startswith("-"):
                    lines.append("+" + line[1:])
                elif line.startswith(" "):
                    lines.append(line)

            return "\n".join(lines)
        except Exception:
            return None
