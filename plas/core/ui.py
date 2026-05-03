"""Rich terminal UI helpers."""

from __future__ import annotations

from typing import Iterable, Mapping


def get_console() -> object:
    """Return a Rich console when available."""

    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console()


def print_panel(title: str, body: str, style: str = "cyan") -> None:
    """Print a styled panel with a plain-text fallback."""

    console = get_console()
    if console is None:
        print(f"{title}\n{body}")
        return
    from rich.panel import Panel

    console.print(Panel.fit(body, title=title, border_style=style))


def print_table(title: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Print a table using Rich when available."""

    rows = list(rows)
    if not rows:
        print_panel(title, "No rows.")
        return
    console = get_console()
    if console is None:
        print(title)
        for row in rows:
            print(row)
        return
    from rich.table import Table

    table = Table(title=title)
    columns = list(rows[0].keys())
    for column in columns:
        table.add_column(str(column))
    for row in rows:
        table.add_row(*[str(row.get(column, "")) for column in columns])
    console.print(table)


def status(message: str) -> object:
    """Return a Rich status context manager or a no-op fallback."""

    console = get_console()
    if console is None:
        print(message)
        from contextlib import nullcontext

        return nullcontext()
    return console.status(message)
