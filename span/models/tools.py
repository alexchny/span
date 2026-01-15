from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None

    def to_content(self) -> list[dict[str, Any]]:
        if self.success:
            return [{"type": "text", "text": self.output}]
        else:
            error_msg = f"Error: {self.error}\n{self.output}" if self.error else self.output
            return [{"type": "text", "text": error_msg}]


@dataclass
class ApplyPatchResult(ToolResult):
    file_path: str | None = None
    reverse_diff: str | None = None
