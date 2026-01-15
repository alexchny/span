from pathlib import Path
from typing import Any

from span.models.tools import ApplyPatchResult, ToolResult
from span.tools.base import Tool
from span.tools.file_ops import ApplyPatchTool, ReadFileTool
from span.tools.shell import RunShellTool


def test_tool_result_success() -> None:
    result = ToolResult(success=True, output="File read successfully")

    assert result.success is True
    assert result.output == "File read successfully"
    assert result.error is None


def test_tool_result_failure() -> None:
    result = ToolResult(success=False, output="", error="File not found")

    assert result.success is False
    assert result.error == "File not found"


def test_tool_result_to_content_success() -> None:
    result = ToolResult(success=True, output="Hello world")
    content = result.to_content()

    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Hello world"


def test_tool_result_to_content_failure() -> None:
    result = ToolResult(success=False, output="Details here", error="Something failed")
    content = result.to_content()

    assert len(content) == 1
    assert "Error: Something failed" in content[0]["text"]
    assert "Details here" in content[0]["text"]


def test_apply_patch_result() -> None:
    result = ApplyPatchResult(
        success=True,
        output="Patch applied",
        file_path="test.py",
        reverse_diff="--- a/test.py\n+++ b/test.py",
    )

    assert result.success is True
    assert result.file_path == "test.py"
    assert result.reverse_diff is not None


class MockTool(Tool):
    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "input": {
                "type": "string",
                "description": "Input string",
                "required": True,
            },
            "optional": {
                "type": "boolean",
                "description": "Optional flag",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="Executed")


def test_tool_to_anthropic_schema() -> None:
    tool = MockTool()
    schema = tool.to_anthropic_tool()

    assert schema["name"] == "mock_tool"
    assert schema["description"] == "A mock tool for testing"
    assert schema["input_schema"]["type"] == "object"
    assert "input" in schema["input_schema"]["properties"]
    assert "optional" in schema["input_schema"]["properties"]
    assert "input" in schema["input_schema"]["required"]
    assert "optional" not in schema["input_schema"]["required"]


def test_read_file_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("line 1\nline 2\nline 3")

    tool = ReadFileTool()
    result = tool.execute(path=str(test_file))

    assert result.success is True
    assert "1|line 1" in result.output
    assert "2|line 2" in result.output
    assert "3|line 3" in result.output


def test_read_file_not_found(tmp_path: Path) -> None:
    tool = ReadFileTool()
    result = tool.execute(path=str(tmp_path / "nonexistent.txt"))

    assert result.success is False
    assert "File not found" in result.error


def test_read_file_is_directory(tmp_path: Path) -> None:
    tool = ReadFileTool()
    result = tool.execute(path=str(tmp_path))

    assert result.success is False
    assert "not a file" in result.error


def test_apply_patch_lazy_pattern_detection() -> None:
    tool = ApplyPatchTool()

    lazy_patch = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
 def foo():
-    pass
+    print("hello")
+    ... rest of code
"""

    assert not tool._validate_patch(lazy_patch)


def test_apply_patch_insufficient_context() -> None:
    tool = ApplyPatchTool()

    no_context_patch = """--- a/test.py
+++ b/test.py
@@ -1,2 +1,2 @@
-old line
+new line
"""

    assert not tool._validate_patch(no_context_patch)


def test_apply_patch_sufficient_context_before() -> None:
    tool = ApplyPatchTool()

    patch = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
 line 1
 line 2
 line 3
-old line
+new line
"""

    assert tool._validate_patch(patch)


def test_apply_patch_sufficient_context_after() -> None:
    tool = ApplyPatchTool()

    patch = """--- a/test.py
+++ b/test.py
@@ -1,5 +1,5 @@
-old line
+new line
 line 1
 line 2
 line 3
"""

    assert tool._validate_patch(patch)


def test_apply_patch_extract_file_path() -> None:
    tool = ApplyPatchTool()

    patch = """--- a/src/test.py
+++ b/src/test.py
@@ -1,2 +1,2 @@
 content
"""

    file_path = tool._extract_file_path(patch)
    assert file_path == Path("src/test.py")


def test_apply_patch_success(tmp_path: Path) -> None:
    test_file = tmp_path / "test.py"
    test_file.write_text("line 0\nline 1\nline 2\nold line\nline 3\nline 4\nline 5\n")

    import os

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)

        patch = """--- test.py
+++ test.py
@@ -1,7 +1,7 @@
 line 0
 line 1
 line 2
-old line
+new line
 line 3
 line 4
 line 5
"""

        tool = ApplyPatchTool()
        result = tool.execute(patch=patch)

        assert result.success is True
        assert result.file_path == "test.py"
        assert result.reverse_diff is not None
        assert "new line" in test_file.read_text()
    finally:
        os.chdir(original_cwd)


def test_run_shell_allowed_pytest(tmp_path: Path) -> None:
    tool = RunShellTool()
    result = tool.execute(command="pytest --version")

    assert result.success is True
    assert "pytest" in result.output.lower()


def test_run_shell_disallowed_program() -> None:
    tool = RunShellTool()
    result = tool.execute(command="rm -rf /")

    assert result.success is False
    assert "not allowed" in result.error


def test_run_shell_disallowed_flag() -> None:
    tool = RunShellTool()
    result = tool.execute(command="pytest --rootdir=/etc")

    assert result.success is False
    assert "Flag not allowed" in result.error


def test_run_shell_path_traversal() -> None:
    tool = RunShellTool()
    result = tool.execute(command="pytest ../../../etc/passwd")

    assert result.success is False
    assert "Suspicious path" in result.error


def test_run_shell_absolute_path() -> None:
    tool = RunShellTool()
    result = tool.execute(command="pytest /etc/passwd")

    assert result.success is False
    assert "Suspicious path" in result.error


def test_run_shell_git_allowed() -> None:
    tool = RunShellTool()
    result = tool.execute(command="git status")

    assert result.success in (True, False)


def test_run_shell_git_disallowed_positional() -> None:
    tool = RunShellTool()
    result = tool.execute(command="git log some_file.txt")

    assert result.success is False
    assert "Positional arguments not allowed" in result.error


def test_run_shell_parse_error() -> None:
    tool = RunShellTool()
    result = tool.execute(command='pytest "unclosed')

    assert result.success is False
    assert "Failed to parse" in result.error
