import os
from typing import Any

import click

from span.config import load_config
from span.context.repo_map import RepoMap
from span.core.agent import Agent, RevertError
from span.core.verifier import Verifier
from span.events.stream import EventStream
from span.llm.client import LLMClient


@click.group()
def cli() -> None:
    pass


@cli.command(name="run")
@click.argument("task")
@click.option("--plan", is_flag=True, help="Show plan for approval before executing")
@click.option("--opus", is_flag=True, help="Use claude-3-opus instead of sonnet")
@click.option("--full", is_flag=True, help="Run full test suite instead of smart selection")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed LLM responses")
def run_task(task: str, plan: bool, opus: bool, full: bool, verbose: bool) -> None:
    try:
        config = load_config()
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        raise click.Abort() from e

    if not config.api_key:
        click.echo("Error: ANTHROPIC_API_KEY not found in environment", err=True)
        click.echo("Set it with: export ANTHROPIC_API_KEY=your-key", err=True)
        raise click.Abort() from None

    if opus:
        config.model = "claude-3-opus-20240229"

    if full:
        config.verification.pytest = True

    if verbose:
        os.environ["SPAN_VERBOSE"] = "1"

    try:
        repo_map = RepoMap()
        llm_client = LLMClient(model=config.model, api_key=config.api_key)
        verifier = Verifier(
            repo_map=repo_map,
            test_patterns=config.test_patterns,
            fallback_tests=config.fallback_tests,
        )
        event_stream = EventStream()
        agent = Agent(config, repo_map, llm_client, verifier, event_stream)

        state = agent.run(task, show_plan=plan)

        if state.changes:
            keep_changes = agent.finalize(state)

            if not keep_changes:
                revision = input("\nRevise instruction (or press Enter to exit): ").strip()
                if revision:
                    new_state = agent.handle_revision(state, revision, show_plan=plan)
                    if new_state.changes:
                        agent.finalize(new_state)
                    else:
                        click.echo("\nNo successful changes.")
        else:
            click.echo("\nNo successful changes.")

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted by user", err=True)
        raise click.Abort() from None
    except RevertError as e:
        click.echo(f"\nFailed to revert changes: {e}", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        raise click.Abort() from e
    finally:
        if "repo_map" in locals():
            repo_map.close()


@cli.command()
def status() -> None:
    event_stream = EventStream()
    events = event_stream.read_all()

    if not events:
        click.echo("No sessions found.")
        return

    sessions: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event_type == "session_start":
            session_id = event.data.get("session_id")
            if session_id:
                sessions[session_id] = {
                    "task": event.data.get("task"),
                    "changes": 0,
                    "errors": [],
                    "timestamp": event.timestamp,
                }
        elif event.event_type == "tool_result":
            session_id = event.data.get("session_id")
            if session_id in sessions:
                result = event.data.get("result", [])
                if isinstance(result, list) and result:
                    text = result[0].get("text", "")
                    if "applied and verified" in text.lower():
                        sessions[session_id]["changes"] += 1
                    elif "error" in text.lower():
                        sessions[session_id]["errors"].append(text[:80])

    if sessions:
        last_session_id = list(sessions.keys())[-1]
        last_session = sessions[last_session_id]

        click.echo(f"Last session: {last_session_id}")
        click.echo(f"Task: {last_session['task']}")
        click.echo(f"Changes: {last_session['changes']}")
        if last_session["errors"]:
            click.echo(f"Errors: {len(last_session['errors'])}")
            for err in last_session["errors"][:3]:
                click.echo(f"  - {err}")
    else:
        click.echo("No session data found.")


@cli.command()
@click.option("--session", help="Filter by session ID")
@click.option("--tail", type=int, help="Show last N events")
def logs(session: str | None, tail: int | None) -> None:
    event_stream = EventStream()
    events = event_stream.read_all()

    if not events:
        click.echo("No events found.")
        return

    if session:
        events = [e for e in events if e.data.get("session_id") == session]

    if tail:
        events = events[-tail:]

    for event in events:
        click.echo(f"[{event.timestamp}] {event.event_type}")
        for key, value in event.data.items():
            if key != "result":
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                click.echo(f"  {key}: {value_str}")


@cli.command()
@click.option("--session", help="Show diff for specific session")
def diff(session: str | None) -> None:
    event_stream = EventStream()
    events = event_stream.read_all()

    if not events:
        click.echo("No events found.")
        return

    if session:
        events = [e for e in events if e.data.get("session_id") == session]

    changes = []
    for event in events:
        if event.event_type == "tool_call" and event.data.get("tool") == "apply_patch":
            path = event.data.get("args", {}).get("path")
            diff = event.data.get("args", {}).get("diff")
            if path and diff:
                changes.append((path, diff))

    if not changes:
        click.echo("No changes found.")
        return

    click.echo("=" * 60)
    click.echo("CHANGES")
    click.echo("=" * 60)

    for path, diff_content in changes:
        click.echo(f"\n{path}")
        click.echo("-" * 60)
        click.echo(diff_content)


if __name__ == "__main__":
    cli()
