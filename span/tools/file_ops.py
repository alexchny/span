import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from span.models.tools import ApplyPatchResult, ToolResult
from span.tools.base import Tool

logger = logging.getLogger(__name__)


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
            "path": {
                "type": "string",
                "description": "Path to the file to patch",
                "required": True,
            },
            "diff": {
                "type": "string",
                "description": "Unified diff content (without file headers). Must include ≥3 context lines before OR after changes.",
                "required": True,
            },
        }

    def execute(self, **kwargs: Any) -> ApplyPatchResult:
        file_path_str = kwargs["path"]
        diff_content = kwargs["diff"]
        file_path = Path(file_path_str)

        validation_error = self._validate_patch_with_reason(diff_content)
        if validation_error:
            return ApplyPatchResult(
                success=False,
                output="",
                error=f"Invalid patch format: {validation_error}. Use proper unified diff with @@ hunk headers and +/- line prefixes.",
            )

        if diff_content.strip().startswith("---") or diff_content.strip().startswith("diff --git"):
            full_patch = diff_content
        else:
            full_patch = f"--- {file_path_str}\n+++ {file_path_str}\n{diff_content}"

        reverse_diff = self._generate_reverse_diff(file_path, full_patch)

        strip_level = "1" if ("--- a/" in full_patch or "+++ b/" in full_patch) else "0"

        result = subprocess.run(
            ["patch", f"-p{strip_level}"],
            input=full_patch.encode(),
            capture_output=True,
        )

        if result.returncode == 0:
            return ApplyPatchResult(
                success=True,
                output=f"Patch applied successfully to {file_path_str}",
                file_path=file_path_str,
                reverse_diff=reverse_diff,
            )
        else:
            error_output = result.stderr.decode() if result.stderr else "Unknown error"
            stdout_output = result.stdout.decode() if result.stdout else ""
            line_count = len(file_path.read_text().splitlines()) if file_path.exists() else 0
            hint = f" (file has {line_count} lines)" if "No such line" in stdout_output else ""
            return ApplyPatchResult(
                success=False,
                output=f"{error_output}\n{stdout_output}".strip(),
                error=f"Patch failed{hint}: {error_output or stdout_output}",
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
        return self._validate_patch_with_reason(patch) is None

    def _validate_patch_with_reason(self, patch: str) -> str | None:
        for pattern in self.LAZY_PATTERNS:
            if re.search(pattern, patch, re.IGNORECASE):
                return "contains lazy placeholder pattern"

        if "@@" not in patch:
            return "missing @@ hunk header"

        hunks = self._extract_hunks(patch)
        if not hunks:
            return "no valid hunks found"

        for hunk in hunks:
            if not self._is_well_formed_hunk(hunk):
                return "lines must start with space, +, or -"
            if not self._has_sufficient_context(hunk):
                return "insufficient context lines"

        return None

    def _is_well_formed_hunk(self, hunk: str) -> bool:
        lines = hunk.split("\n")
        if not lines:
            return False

        if not lines[0].startswith("@@"):
            return False

        for line in lines[1:]:
            if not line:
                continue
            if line[0] not in (" ", "+", "-", "\\"):
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
        hunk_header = hunk.split("\n")[0]
        if hunk_header.startswith("@@"):
            if "-0,0" in hunk_header or "@@ -0,0" in hunk_header:
                return True

        lines = hunk.split("\n")[1:]
        context_before = 0
        context_after = 0
        seen_change = False
        has_deletions = False

        for line in lines:
            if line.startswith(" "):
                if not seen_change:
                    context_before += 1
                else:
                    context_after += 1
            elif line.startswith("-"):
                seen_change = True
                has_deletions = True
                context_after = 0
            elif line.startswith("+"):
                seen_change = True
                context_after = 0

        if context_before >= 3 or context_after >= 3:
            return True

        if not has_deletions and context_before >= 1:
            if context_before < 3:
                logger.warning(
                    "Accepting append-only patch with minimal context (%d line(s)). "
                    "This may cause incorrect edits in repetitive code.",
                    context_before,
                )
            return True

        return False

    def _generate_reverse_diff(self, file_path: Path, patch: str) -> str | None:
        if not file_path.exists():
            return None

        try:
            lines = []
            lines.append(f"--- {file_path}")
            lines.append(f"+++ {file_path}")

            for line in patch.split("\n"):
                if line.startswith("--- ") or line.startswith("+++ "):
                    continue
                if line.startswith("diff ") or line.startswith("index "):
                    continue
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
