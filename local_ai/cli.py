"""local-ai command-line interface.

Commands: index, review, ask, plan, patch, apply, diff, run, config.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import context_builder as ctx
from . import patcher, prompts, report_writer
from . import scanner as scan
from .command_runner import CommandResult, check_dangerous, run_command, tail
from .config import Config, ConfigError, ensure_default_config, load_config
from .memory import Memory
from .model_client import ModelClient, ModelError
from .types import RepoIndex

app = typer.Typer(
    name="local-ai",
    help="Local-first coding assistant for LM Studio. Scans a repo, builds context, "
    "and asks your local model for reviews, answers, plans, and patches.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- helpers


def _resolve_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        err_console.print(f"[red]Error:[/red] '{path}' is not a directory.")
        raise typer.Exit(code=2)
    return root


def _load_config_or_exit(root: Path) -> Config:
    try:
        return load_config(root)
    except ConfigError as exc:
        err_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=2)


def _index_to_json(index: RepoIndex) -> str:
    payload = {
        "root": str(index.root),
        "file_count": index.file_count,
        "total_size_bytes": index.total_size_bytes,
        "frameworks": index.frameworks,
        "languages": index.languages,
        "important_files": index.important_files,
        "files": [
            {
                "path": f.rel_path,
                "size_bytes": f.size_bytes,
                "language": f.language,
                "lines": f.lines,
                "is_important": f.is_important,
                "is_binary": f.is_binary,
                "imports": f.imports,
                "symbols": f.symbols,
            }
            for f in index.files
        ],
    }
    return json.dumps(payload, indent=2)


def _do_index(root: Path, *, quiet: bool = False) -> RepoIndex:
    if not quiet:
        with console.status("[cyan]Scanning repository...[/cyan]"):
            index = scan.scan_repo(root)
    else:
        index = scan.scan_repo(root)
    repo_map = ctx.build_repo_map(index)
    report_writer.write_repo_map(root, repo_map)
    report_writer.write_repo_index(root, _index_to_json(index))
    mem = Memory.load(root)
    mem.note_index(index.file_count, index.frameworks)
    mem.record("index", file_count=index.file_count)
    mem.save()
    return index


def _get_index(root: Path) -> RepoIndex:
    """Always rescan (fast) so context reflects current disk state."""
    return _do_index(root, quiet=True)


def _call_model(config: Config, system: str, context_body: str, request: str) -> str:
    client = ModelClient(config)
    messages = prompts.build_messages(system, context_body, request)
    try:
        with console.status(f"[cyan]Asking {config.model}...[/cyan]"):
            response = client.chat(messages)
    except ModelError as exc:
        err_console.print(Panel.fit(str(exc), title="[red]Model error[/red]", border_style="red"))
        if exc.hint:
            err_console.print(f"[yellow]Hint:[/yellow] {exc.hint}")
        raise typer.Exit(code=1)
    if response.prompt_tokens:
        console.print(
            f"[dim]tokens: prompt={response.prompt_tokens} "
            f"completion={response.completion_tokens} "
            f"finish={response.finish_reason}[/dim]"
        )
    return response.content


def _print_context_summary(packet: ctx.ContextPacket) -> None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[dim]context tokens (approx)[/dim]", f"~{packet.approx_tokens}")
    table.add_row("[dim]files included[/dim]", str(len(packet.included_files)))
    if packet.skipped_files:
        table.add_row("[dim]files dropped (budget)[/dim]", str(len(packet.skipped_files)))
    console.print(table)
    if packet.included_files:
        console.print("[dim]" + ", ".join(packet.included_files[:12]) + (
            " ..." if len(packet.included_files) > 12 else ""
        ) + "[/dim]")


# --------------------------------------------------------------------------- commands


@app.command()
def index(path: str = typer.Argument(..., help="Path to the repository to scan.")) -> None:
    """Scan a repo and write .local-ai/repo_map.md + repo_index.json."""
    root = _resolve_root(path)
    ensure_default_config(root)
    index = _do_index(root)

    table = Table(title=f"Indexed {root.name}", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Files scanned", str(index.file_count))
    table.add_row("Total size", f"{index.total_size_bytes / 1024:.1f} KB")
    table.add_row("Detected stack", ", ".join(index.frameworks) or "—")
    table.add_row("Languages", ", ".join(f"{k}({v})" for k, v in index.languages.items()) or "—")
    table.add_row("Important files", str(len(index.important_files)))
    console.print(table)
    console.print(f"[green]✓[/green] Wrote [bold].local-ai/repo_map.md[/bold] and "
                  f"[bold].local-ai/repo_index.json[/bold]")


@app.command()
def review(path: str = typer.Argument(..., help="Path to the repository to review.")) -> None:
    """Generate a full repository review report."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    index = _get_index(root)

    repo_map = ctx.build_repo_map(index)
    query = "overall architecture entry points bugs security performance dependencies tests"
    packet = ctx.build_context(index, query, max_context_tokens=config.max_context_tokens,
                               repo_map=repo_map)
    _print_context_summary(packet)

    request = (
        "Review this repository thoroughly following the required section structure."
    )
    output = _call_model(config, prompts.REPO_REVIEW_SYSTEM, packet.body, request)

    ts = report_writer.timestamp()
    full = output + report_writer.context_footer(
        packet.included_files, packet.skipped_files, packet.approx_tokens
    )
    report_path = report_writer.write_report(root, full, ts)

    mem = Memory.load(root)
    mem.record("review", report=str(report_path.relative_to(root)))
    mem.save()

    console.print(Panel(output[:1600] + ("\n\n[dim]... (truncated; see full report)[/dim]"
                  if len(output) > 1600 else ""), title="Review summary", border_style="cyan"))
    console.print(f"[green]✓[/green] Full report: [bold]{report_path}[/bold]")


@app.command()
def ask(
    path: str = typer.Argument(..., help="Path to the repository."),
    question: str = typer.Argument(..., help="Your question about the repo."),
) -> None:
    """Answer a question about the repo, selecting relevant files automatically."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    index = _get_index(root)

    packet = ctx.build_context(index, question, max_context_tokens=config.max_context_tokens)
    _print_context_summary(packet)

    output = _call_model(config, prompts.ASK_SYSTEM, packet.body, question)

    mem = Memory.load(root)
    mem.record("ask", question=question[:200])
    mem.save()

    console.print(Panel(output, title="Answer", border_style="cyan"))


@app.command()
def plan(
    path: str = typer.Argument(..., help="Path to the repository."),
    task: str = typer.Argument(..., help="The task to plan."),
) -> None:
    """Produce an implementation plan (no edits)."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    index = _get_index(root)

    packet = ctx.build_context(index, task, max_context_tokens=config.max_context_tokens)
    _print_context_summary(packet)

    output = _call_model(config, prompts.PLAN_SYSTEM, packet.body, f"Task: {task}")

    ts = report_writer.timestamp()
    full = f"# Plan\n\n**Task:** {task}\n\n" + output + report_writer.context_footer(
        packet.included_files, packet.skipped_files, packet.approx_tokens
    )
    plan_path = report_writer.write_plan(root, full, ts)

    mem = Memory.load(root)
    mem.record("plan", task=task[:200], plan=str(plan_path.relative_to(root)))
    mem.save()

    console.print(Panel(output, title="Implementation plan", border_style="cyan"))
    console.print(f"[green]✓[/green] Plan written: [bold]{plan_path}[/bold]")


@app.command()
def patch(
    path: str = typer.Argument(..., help="Path to the repository."),
    task: str = typer.Argument(..., help="The change to propose as a diff."),
) -> None:
    """Propose a unified diff for a task (does NOT apply it)."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)
    index = _get_index(root)

    packet = ctx.build_context(index, task, max_context_tokens=config.max_context_tokens)
    _print_context_summary(packet)

    output = _call_model(config, prompts.PATCH_SYSTEM, packet.body, f"Task: {task}")

    diff_text = patcher.extract_diff(output)
    ts = report_writer.timestamp()

    explanation = f"# Patch proposal\n\n**Task:** {task}\n\n" + output
    explanation += report_writer.context_footer(
        packet.included_files, packet.skipped_files, packet.approx_tokens
    )
    diff_path, md_path = report_writer.write_patch(root, diff_text, explanation, ts)

    mem = Memory.load(root)
    mem.record("patch", task=task[:200], produced_diff=bool(diff_text))
    mem.save()

    console.print(Panel(output[:1600] + ("\n\n[dim]...[/dim]" if len(output) > 1600 else ""),
                  title="Patch explanation", border_style="cyan"))

    if diff_text:
        stats = patcher.diff_stats(diff_text)
        console.print(f"[green]✓[/green] Proposed diff: [bold]{diff_path}[/bold]")
        console.print(
            f"[dim]{len(stats.files)} file(s), +{stats.additions}/-{stats.deletions}[/dim]"
        )
        # Best-effort cleanliness check.
        check = patcher.check_apply(root, diff_path)  # type: ignore[arg-type]
        if check.ok:
            console.print("[green]✓[/green] Patch applies cleanly (git apply --check).")
        else:
            console.print(f"[yellow]⚠ Patch may not apply cleanly:[/yellow] {check.message}")
        console.print(f"[dim]Apply with:[/dim] local-ai apply {path} {diff_path}")
    else:
        console.print("[yellow]No valid diff produced — wrote a plan/explanation instead.[/yellow]")
    console.print(f"[green]✓[/green] Explanation: [bold]{md_path}[/bold]")


@app.command()
def apply(
    path: str = typer.Argument(..., help="Path to the repository."),
    patch_file: str = typer.Argument(..., help="Path to the .diff file to apply."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Apply a previously generated patch (with confirmation)."""
    root = _resolve_root(path)
    diff_path = Path(patch_file).expanduser().resolve()
    if not diff_path.is_file():
        err_console.print(f"[red]Error:[/red] patch file '{patch_file}' not found.")
        raise typer.Exit(code=2)

    diff_text = diff_path.read_text(encoding="utf-8")
    stats = patcher.diff_stats(diff_text)

    table = Table(title="Patch to apply", header_style="bold yellow")
    table.add_column("File")
    for f in stats.files:
        table.add_row(f)
    console.print(table)
    console.print(f"[bold]+{stats.additions}[/bold] additions, "
                  f"[bold]-{stats.deletions}[/bold] deletions across "
                  f"{len(stats.files)} file(s).")

    check = patcher.check_apply(root, diff_path)
    if not check.ok:
        err_console.print(f"[red]Patch does not apply cleanly:[/red] {check.message}")
        err_console.print("[red]Aborting — no files were modified.[/red]")
        raise typer.Exit(code=1)

    console.print("[yellow]⚠ This will modify files in your working tree.[/yellow]")
    if not yes:
        confirmed = typer.confirm("Apply this patch?", default=False)
        if not confirmed:
            console.print("Aborted. No changes made.")
            raise typer.Exit(code=0)

    result = patcher.apply_patch(root, diff_path)
    if result.ok:
        mem = Memory.load(root)
        mem.record("apply", patch=diff_path.name)
        mem.save()
        console.print(f"[green]✓ {result.message}[/green]")
    else:
        err_console.print(f"[red]git apply failed:[/red] {result.message}")
        err_console.print("[red]No partial changes were made.[/red]")
        raise typer.Exit(code=1)


@app.command()
def diff(path: str = typer.Argument(..., help="Path to the git repository.")) -> None:
    """Review the current git diff with the model."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)

    if not patcher.git_available():
        err_console.print("[red]git is not installed; cannot read a diff.[/red]")
        raise typer.Exit(code=2)

    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD"], cwd=root, capture_output=True, text=True
        )
    except OSError as exc:
        err_console.print(f"[red]Failed to run git diff:[/red] {exc}")
        raise typer.Exit(code=2)

    if proc.returncode != 0:
        err_console.print(f"[red]git diff failed:[/red] {proc.stderr.strip()}")
        err_console.print("[yellow]Is this a git repository with commits?[/yellow]")
        raise typer.Exit(code=1)

    diff_text = proc.stdout
    if not diff_text.strip():
        console.print("[yellow]No changes detected (git diff HEAD is empty).[/yellow]")
        raise typer.Exit(code=0)

    # Keep the diff within budget (diffs can be large).
    max_chars = config.max_context_tokens * ctx.CHARS_PER_TOKEN
    if len(diff_text) > max_chars:
        diff_text = diff_text[:max_chars] + "\n...[diff truncated for context budget]...\n"

    context_body = "## Current git diff (HEAD)\n\n```diff\n" + diff_text + "\n```\n"
    output = _call_model(config, prompts.DIFF_REVIEW_SYSTEM, context_body,
                         "Review this diff.")

    mem = Memory.load(root)
    mem.record("diff_review")
    mem.save()
    console.print(Panel(output, title="Diff review", border_style="cyan"))


@app.command()
def run(
    path: str = typer.Argument(..., help="Path to run the command from."),
    command: str = typer.Argument(..., help="Shell command to run."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip dangerous-command confirmation."),
) -> None:
    """Run a shell command; on failure, ask the model to diagnose it."""
    root = _resolve_root(path)
    ensure_default_config(root)
    config = _load_config_or_exit(root)

    dangers = check_dangerous(command)
    if dangers and not yes:
        err_console.print(Panel(
            f"This command looks potentially destructive:\n  [bold]{command}[/bold]\n\n"
            f"Matched: {', '.join(dangers)}",
            title="[red]⚠ Dangerous command[/red]", border_style="red",
        ))
        if not typer.confirm("Run it anyway?", default=False):
            console.print("Aborted.")
            raise typer.Exit(code=0)

    console.print(f"[cyan]$ {command}[/cyan]  [dim](in {root})[/dim]")
    result = run_command(command, root)

    if result.stdout:
        console.print(result.stdout.rstrip())
    if result.stderr:
        err_console.print(result.stderr.rstrip())

    status = "[green]exit 0[/green]" if result.ok else f"[red]exit {result.exit_code}[/red]"
    console.print(f"Command finished: {status}")

    ts = report_writer.timestamp()
    report = _build_run_report(result)

    if not result.ok:
        console.print("[yellow]Command failed — asking the model to diagnose...[/yellow]")
        index = _get_index(root)
        # Light context: repo map + the failing command output.
        repo_map = ctx.build_repo_map(index)
        diag_context = (
            repo_map
            + "\n\n## Failed command\n\n```\n" + command + "\n```\n"
            + f"\nExit code: {result.exit_code}\n"
            + "\n## stdout (tail)\n\n```\n" + tail(result.stdout) + "\n```\n"
            + "\n## stderr (tail)\n\n```\n" + tail(result.stderr) + "\n```\n"
        )
        diagnosis = _call_model(
            config, prompts.RUN_DIAGNOSIS_SYSTEM, diag_context,
            "Diagnose why this command failed and how to fix it.",
        )
        report += "\n\n## Diagnosis\n\n" + diagnosis
        console.print(Panel(diagnosis, title="Diagnosis", border_style="yellow"))

    run_path = report_writer.write_run(root, report, ts)
    mem = Memory.load(root)
    mem.record("run", command=command[:200], exit_code=result.exit_code)
    mem.save()
    console.print(f"[green]✓[/green] Run report: [bold]{run_path}[/bold]")
    if not result.ok:
        raise typer.Exit(code=result.exit_code)


@app.command()
def config(path: str = typer.Argument(".", help="Path to the repository.")) -> None:
    """Show the effective configuration for a repo (and create defaults if missing)."""
    root = _resolve_root(path)
    cfg_path = ensure_default_config(root)
    cfg = _load_config_or_exit(root)
    table = Table(title="Effective config", header_style="bold cyan")
    table.add_column("Key")
    table.add_column("Value")
    for k, v in cfg.to_dict().items():
        masked = "***" if k == "api_key" else str(v)
        table.add_row(k, masked)
    console.print(table)
    console.print(f"[dim]Config file: {cfg_path}[/dim]")
    # Quick reachability probe.
    client = ModelClient(cfg)
    reachable = client.health_check()
    if reachable:
        console.print(f"[green]✓[/green] LM Studio reachable at {cfg.base_url}")
    else:
        console.print(f"[yellow]⚠ Could not reach {cfg.base_url}/models — "
                      f"is the LM Studio server started?[/yellow]")


def _build_run_report(result: CommandResult) -> str:
    parts = [
        "# Run report",
        "",
        f"**Command:** `{result.command}`",
        f"**Working dir:** `{result.cwd}`",
        f"**Exit code:** {result.exit_code}"
        + ("  (timed out)" if result.timed_out else ""),
        "",
        "## stdout",
        "",
        "```",
        result.stdout.rstrip() or "(empty)",
        "```",
        "",
        "## stderr",
        "",
        "```",
        result.stderr.rstrip() or "(empty)",
        "```",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    app()
