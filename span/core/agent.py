import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from span.config import Config
from span.context.repo_map import RepoMap
from span.core.verifier import Verifier
from span.events.stream import EventStream
from span.llm.client import LLMClient
from span.llm.prompts import EXECUTE_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT
from span.tools.file_ops import ApplyPatchTool, ReadFileTool
from span.tools.shell import RunShellTool


class RevertError(Exception):
    def __init__(self, failed_ops: list[tuple[str, str, str]]):
        self.failed_ops = failed_ops
        paths = [op[0] for op in failed_ops]
        super().__init__(f"Failed to revert changes in: {', '.join(paths)}")


@dataclass
class ChangeOp:
    path: str
    forward_diff: str
    reverse_diff: str
    timestamp: float
    step_id: int


@dataclass
class AgentLimits:
    max_turns: int = 20
    max_tool_calls: int = 50
    max_patch_attempts: int = 15
    max_retries_per_patch: int = 3


@dataclass
class AgentState:
    session_id: str
    messages: list[dict]
    changes: list[ChangeOp] = field(default_factory=list)
    turn_count: int = 0
    tool_call_count: int = 0
    patch_attempt_count: int = 0
    last_errors: list[str] = field(default_factory=list)
    original_task: str = ""
    _retry_count: dict = field(default_factory=dict)
    _created_files: set = field(default_factory=set)


class Agent:
    def __init__(
        self,
        config: Config,
        repo_map: RepoMap,
        llm_client: LLMClient,
        verifier: Verifier,
        event_stream: EventStream,
    ):
        self.config = config
        self.repo_map = repo_map
        self.llm_client = llm_client
        self.verifier = verifier
        self.event_stream = event_stream
        self.limits = AgentLimits(
            max_turns=config.max_steps,
            max_retries_per_patch=config.max_retries_per_step,
        )

        self.read_file_tool = ReadFileTool()
        self.apply_patch_tool = ApplyPatchTool()
        self.run_shell_tool = RunShellTool()

    def run(self, task: str, show_plan: bool = False) -> AgentState:
        session_id = self._generate_session_id()

        print("Planning...")
        plan = self._get_plan(task, session_id)

        preview = self._format_plan_preview(plan)
        print(f"\n{preview}\n")

        if show_plan:
            response = input("Proceed? [Y/n]: ").strip().lower()
            if response == "n":
                return AgentState(session_id=session_id, messages=[], original_task=task)

        exec_messages = [
            {"role": "user", "content": f"Task: {task}\n\nPlan:\n{plan}"}
        ]

        state = AgentState(
            session_id=session_id,
            messages=exec_messages,
            original_task=task,
        )

        self._execute_loop(state)

        return state

    def continue_turn(
        self, state: AgentState, message: str, plan_mode: bool = False
    ) -> AgentState:
        state.messages.append({"role": "user", "content": message})

        if plan_mode:
            plan = self._get_plan_only(state)
            print(f"\n{plan}\n")
            response = input("Execute this plan? [y/N]: ").strip().lower()
            if response != "y":
                state.messages.append(
                    {"role": "assistant", "content": [{"type": "text", "text": plan}]}
                )
                return state

        self._execute_loop(state)
        return state

    def _get_plan_only(self, state: AgentState) -> str:
        last_message = state.messages[-1]["content"]

        response = self.llm_client.send_message(
            system=PLAN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": last_message}],
            tools=[],
        )

        return self.llm_client.extract_text(response)

    def chat(
        self,
        state: AgentState,
        message: str,
        thinking_mode: str = "off",
        renderer: Any = None,
    ) -> str:
        state.messages.append({"role": "user", "content": message})

        tools = [
            self.read_file_tool.to_anthropic_tool(),
            self.apply_patch_tool.to_anthropic_tool(),
            self.run_shell_tool.to_anthropic_tool(),
        ]

        response = self.llm_client.send_message(
            system=EXECUTE_SYSTEM_PROMPT,
            messages=state.messages,
            tools=tools,
        )

        state.messages.append({"role": "assistant", "content": response.content})

        response_text = ""
        tool_results = []
        had_successful_patch = False

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if renderer:
                    renderer.render_tool_call(tool_name, tool_input)

                tool_call = {"name": tool_name, "input": tool_input}
                result = self._execute_tool(tool_call, state)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                if renderer:
                    renderer.render_tool_result(tool_name, result)

                if tool_name == "apply_patch":
                    for item in result:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text.startswith("SUCCESS:"):
                                had_successful_patch = True
                                break

        if had_successful_patch:
            state.messages.append({"role": "user", "content": tool_results})
            return response_text or "I've made the changes. Please review."

        if tool_results:
            state.messages.append({"role": "user", "content": tool_results})

            follow_up = self.llm_client.send_message(
                system=EXECUTE_SYSTEM_PROMPT,
                messages=state.messages,
                tools=tools,
            )

            state.messages.append({"role": "assistant", "content": follow_up.content})

            return self._continue_chat_loop(state, follow_up, tools, renderer, thinking_mode, depth=1)

        return response_text

    def _continue_chat_loop(
        self,
        state: AgentState,
        response: Any,
        tools: list[dict],
        renderer: Any,
        thinking_mode: str,
        depth: int = 0,
    ) -> str:
        if depth > 30:
            return "[Reached tool call limit - try breaking your request into smaller steps]"

        response_text = ""
        tool_results = []
        had_successful_patch = False

        for block in response.content:
            if block.type == "text":
                response_text += block.text
            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                if renderer:
                    renderer.render_tool_call(tool_name, tool_input)

                tool_call = {"name": tool_name, "input": tool_input}
                result = self._execute_tool(tool_call, state)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

                if renderer:
                    renderer.render_tool_result(tool_name, result)

                if tool_name == "apply_patch":
                    for item in result:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text.startswith("SUCCESS:"):
                                had_successful_patch = True
                                break

        if had_successful_patch:
            state.messages.append({"role": "user", "content": tool_results})
            return response_text or "I've made the changes. Please review."

        if tool_results:
            state.messages.append({"role": "user", "content": tool_results})

            follow_up = self.llm_client.send_message(
                system=EXECUTE_SYSTEM_PROMPT,
                messages=state.messages,
                tools=tools,
            )

            state.messages.append({"role": "assistant", "content": follow_up.content})

            return self._continue_chat_loop(state, follow_up, tools, renderer, thinking_mode, depth + 1)

        return response_text

    def _generate_session_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _get_plan(self, task: str, session_id: str) -> str:
        plan_messages = [{"role": "user", "content": task}]

        response = self.llm_client.send_message(
            system=PLAN_SYSTEM_PROMPT,
            messages=plan_messages,
            tools=[],
        )

        plan = self.llm_client.extract_text(response)

        self.event_stream.append(
            "plan",
            session_id=session_id,
            task=task,
            plan=plan,
        )

        return plan

    def _format_plan_preview(self, plan: str) -> str:
        lines: list[str] = []
        for line in plan.split("\n"):
            line = line.strip()
            if not line:
                continue

            clean = line.lstrip("123456)-•*# ").strip()
            clean = clean.replace("**", "").strip()

            if any(line.startswith(prefix) for prefix in ["1)", "2)", "3)", "4)", "5)", "6)", "-", "•", "*", "#"]):
                if clean and len(lines) < 6 and not clean.lower().startswith(("plan", "goal", "approach")):
                    lines.append(f"  • {clean}")
            elif ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    clean = parts[1].strip().replace("**", "").strip()
                    if clean and len(lines) < 6:
                        lines.append(f"  • {clean}")

        if not lines:
            summary = " ".join(plan.split()[:50])
            if len(summary) > 200:
                summary = summary[:197] + "..."
            return f"Plan:\n  • {summary}"

        return "Plan:\n" + "\n".join(lines[:6])

    def _execute_loop(self, state: AgentState) -> None:
        tools = [
            self.read_file_tool.to_anthropic_tool(),
            self.apply_patch_tool.to_anthropic_tool(),
            self.run_shell_tool.to_anthropic_tool(),
        ]

        while True:
            if limit := self._check_limits(state):
                print(f"Stopped: {limit} limit reached")
                break

            state.turn_count += 1

            response = self.llm_client.send_message(
                system=EXECUTE_SYSTEM_PROMPT,
                messages=state.messages,
                tools=tools,
            )

            if not self.llm_client.has_tool_use(response):
                if state.last_errors and not state.changes:
                    print("\nAgent stopped after verification failures.")
                break

            state.messages.append({"role": "assistant", "content": response.content})

            tool_calls = self.llm_client.extract_tool_calls(response)
            tool_results = []
            hit_limit = False
            had_successful_patch = False

            for tool_call in tool_calls:
                state.tool_call_count += 1

                if tool_call["name"] == "apply_patch":
                    state.patch_attempt_count += 1

                if limit := self._check_limits(state):
                    print(f"Stopped: {limit} limit reached")
                    hit_limit = True
                    break

                result = self._execute_tool(tool_call, state)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": result,
                    }
                )

                self.event_stream.append(
                    "tool_call",
                    session_id=state.session_id,
                    tool=tool_call["name"],
                    args=tool_call["input"],
                )

                self.event_stream.append(
                    "tool_result",
                    session_id=state.session_id,
                    result=result,
                )

                if tool_call["name"] == "apply_patch":
                    for item in result:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text.startswith("SUCCESS:"):
                                had_successful_patch = True
                                break
                    if had_successful_patch:
                        break

            if hit_limit or had_successful_patch:
                if tool_results:
                    state.messages.append({"role": "user", "content": tool_results})
                break

            if tool_results:
                state.messages.append({"role": "user", "content": tool_results})

    def _check_limits(self, state: AgentState) -> str | None:
        if state.turn_count >= self.limits.max_turns:
            return "max_turns"
        if state.tool_call_count >= self.limits.max_tool_calls:
            return "max_tool_calls"
        if state.patch_attempt_count >= self.limits.max_patch_attempts:
            return "max_patch_attempts"
        return None

    def _execute_tool(self, tool_call: dict, state: AgentState) -> list[dict[str, Any]]:
        tool_name = tool_call["name"]
        tool_input = tool_call["input"]

        if tool_name == "read_file":
            path = tool_input.get("path", "file")
            print(f"Reading {path}...")
            result = self.read_file_tool.execute(**tool_input)
            return result.to_content()

        elif tool_name == "apply_patch":
            return self._execute_patch_with_verification(tool_input, state)

        elif tool_name == "run_shell":
            cmd = tool_input.get("command", "command")
            print(f"Running {cmd}...")
            result = self.run_shell_tool.execute(**tool_input)
            return result.to_content()

        else:
            return [{"type": "text", "text": f"Error: Unknown tool '{tool_name}'"}]

    def _execute_patch_with_verification(
        self, tool_input: dict, state: AgentState
    ) -> list[dict[str, Any]]:
        path = tool_input["path"]
        diff = tool_input["diff"]

        file_path = self._resolve_to_repo_root(path)
        is_new_file = not file_path.exists()

        retry_count = state._retry_count.get(path, 0)
        max_retries = self.limits.max_retries_per_patch

        if retry_count >= max_retries:
            print(f"  ✗ Max retries ({max_retries}) exceeded for {path}")
            return [
                {
                    "type": "text",
                    "text": f"ERROR: Exceeded {max_retries} retry attempts for {path}. Stop trying to patch this file and either try a completely different approach or report that you cannot complete the task.",
                }
            ]

        if retry_count == 0:
            print(f"Applying patch to {path}...")
        else:
            print(f"  Retrying {path}... ({retry_count + 1}/{max_retries})")

        apply_result = self.apply_patch_tool.execute(path=path, diff=diff)

        if not apply_result.success:
            state._retry_count[path] = retry_count + 1
            error_hint = apply_result.error or "unknown error"
            if "No such line" in (apply_result.output or ""):
                error_hint = "line count mismatch"
            elif "FAILED" in (apply_result.output or ""):
                error_hint = "hunk doesn't match file"
            print(f"  ✗ Patch failed ({error_hint})")
            return [{"type": "text", "text": f"Error: {apply_result.error}"}]

        verification = self.verifier.verify_patch(path, strict=False)

        if verification.passed:
            if apply_result.reverse_diff is None and not is_new_file:
                return [{"type": "text", "text": "Error: Failed to generate reverse diff"}]

            lint_result = self.verifier.check_lint([path])
            if lint_result.passed:
                print("  ✓ Verified")
            else:
                print("  ✓ Applied (with lint warnings)")

            if is_new_file:
                state._created_files.add(path)
            state._retry_count.pop(path, None)

            reverse_diff = apply_result.reverse_diff or ""
            state.changes.append(
                ChangeOp(
                    path=path,
                    forward_diff=diff,
                    reverse_diff=reverse_diff,
                    timestamp=time.time(),
                    step_id=state.patch_attempt_count,
                )
            )

            result_msg = f"SUCCESS: Patch to {path} applied."
            if not lint_result.passed:
                first_error = lint_result.errors[0] if lint_result.errors else ""
                if first_error:
                    short_err = first_error.split('\n')[1] if '\n' in first_error else first_error[:100]
                    result_msg += f" Note: has lint warnings ({short_err})"
            result_msg += " Task complete - stop now and let the user review the changes."

            return [{"type": "text", "text": result_msg}]
        else:
            state._retry_count[path] = retry_count + 1
            first_error = verification.errors[0] if verification.errors else "unknown"
            short_error = first_error[:60] + "..." if len(first_error) > 60 else first_error
            print(f"  ✗ Verification failed: {short_error}")

            if apply_result.original_content is not None and isinstance(apply_result.original_content, str):
                try:
                    file_path.write_text(apply_result.original_content)
                    print("  ↩ Restored original file")
                except OSError as e:
                    self.event_stream.append(
                        "revert_failed",
                        session_id=state.session_id,
                        path=path,
                        error=str(e),
                    )
            elif is_new_file:
                try:
                    file_path.unlink()
                    print("  ↩ Removed failed new file")
                except OSError:
                    pass

            state.last_errors = verification.errors
            error_msg = "\n".join(verification.errors)
            return [
                {
                    "type": "text",
                    "text": f"Patch reverted due to verification failure:\n{error_msg}",
                }
            ]

    def _resolve_to_repo_root(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / "span.yaml").exists() or (parent / ".git").exists():
                return parent / path
        return p.resolve()

    def _apply_reverse_diff(self, path: str, reverse_diff: str) -> bool:
        result = self.apply_patch_tool.execute(path=path, diff=reverse_diff)
        return result.success

    def revert_all(self, changes: list[ChangeOp]) -> None:
        failed_ops: list[tuple[str, str, str]] = []

        for op in reversed(changes):
            success = self._apply_reverse_diff(op.path, op.reverse_diff)
            if not success:
                failed_ops.append((op.path, op.reverse_diff, f"Failed to revert {op.path}"))

        if failed_ops:
            raise RevertError(failed_ops)

    def finalize(self, state: AgentState) -> bool:
        if not state.changes:
            return False

        final_check = self.verifier.verify_final()

        self._show_diff(state.changes, state._created_files)

        if not final_check.passed:
            print("\nNote: Some optional checks have warnings:")
            for err in final_check.errors[:3]:
                if "type annotation" in err.lower() or "no-untyped" in err:
                    print("  - Consider adding type hints")
                    break
                else:
                    short = err[:70] + "..." if len(err) > 70 else err
                    print(f"  - {short}")

        response = input("\nKeep changes? [y/N]: ").strip().lower()

        if response == "y":
            state.changes.clear()
            return True
        else:
            print("Reverting changes...")
            self.revert_all(state.changes)
            return False

    def _show_diff(self, changes: list[ChangeOp], created_files: set | None = None) -> None:
        print("\n" + "─" * 60)
        created = created_files or set()

        for op in changes:
            tag = " (new file)" if op.path in created else ""
            print(f"\n{op.path}{tag}")
            print(op.forward_diff)

    def handle_revision(
        self, state: AgentState, revision: str, show_plan: bool = False
    ) -> AgentState:
        summary = self._build_run_summary(state)

        fresh_task = f"""Previous run summary:
{summary}

User revision: {revision}"""

        return self.run(task=fresh_task, show_plan=show_plan)

    def _build_run_summary(self, state: AgentState) -> str:
        lines = []
        lines.append(f"Original task: {state.original_task}")
        lines.append(f"Steps taken: {state.tool_call_count}")

        if state.changes:
            lines.append("Successful changes:")
            for op in state.changes:
                lines.append(f"  - {op.path}")

        if state.last_errors:
            lines.append("Last errors:")
            for err in state.last_errors[:3]:
                lines.append(f"  - {err}")

        return "\n".join(lines)
