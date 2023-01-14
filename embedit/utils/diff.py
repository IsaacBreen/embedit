import difflib
from typing import NamedTuple, Optional

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textwrap import indent

def diff_strings(a: str, b: str) -> list[str]:
    return list(difflib.Differ().compare(a.splitlines(), b.splitlines()))


class DiffLineNumbers(NamedTuple):
    added: list[int]
    removed: list[int]

def diff_line_numbers(difflines: list[str]) -> DiffLineNumbers:
    """
    Get the line numbers of the added and removed lines in a diff.

    :param diff: The diff to get the line numbers of.
    :return: A named tuple containing the line numbers of the added and removed lines relative to the original and new files respectively.
    """
    added = []
    removed = []
    for i, line in enumerate(difflines):
        if line.startswith("+"):
            added.append(i - len(removed))
        elif line.startswith("-"):
            removed.append(i - len(added))
    return DiffLineNumbers(added=added, removed=removed)


def prettify_diff(difflines: list[str], context: int = 3, syntax: str = "python"):
    # Get the intervals to show
    intervals = []
    for i, line in enumerate(difflines):
        if line.startswith("+") or line.startswith("-"):
            if intervals and intervals[-1][1] >= i - context:
                # Extend the last interval
                intervals[-1][1] = i + context
            else:
                # Start a new interval
                intervals.append([i - context, i + context])
    intervals = [(max(0, start), min(len(difflines), end)) for start, end in intervals]
    # Get the lines to show
    diffline_chunks = {
        interval[0]: difflines[interval[0] : interval[1]]
        for interval in intervals
    }
    panels: list[Panel] = []
    num_lines_removed_so_far = 0
    for start, lines in diffline_chunks.items():
        chunks: list[tuple[str, list[str]]] = []
        chunk = []
        prev_chunk_type = None
        for i, line in enumerate(lines):
            if line.startswith("+ "):
                chunk_type = "add"
            elif line.startswith("  "):
                chunk_type = "equal"
            elif line.startswith("- "):
                chunk_type = "remove"
            else:
                raise ValueError(f"Unexpected line in diff: {line}")
            if chunk_type != prev_chunk_type:
                if chunk:
                    chunks.append((prev_chunk_type, chunk))
                chunk = []
            chunk.append(line[2:])
            prev_chunk_type = chunk_type
        if chunk:
            chunks.append((prev_chunk_type, chunk))
        syntax_chunks = []
        line_number = start - num_lines_removed_so_far
        for chunk_type, chunk in chunks:
            if chunk_type == "add":
                syntax_chunks.append(Syntax("\n".join(chunk), syntax, theme="github-dark", line_numbers=True, start_line=line_number, background_color="rgb(0,80,0)"))
                line_number += len(chunk)
            elif chunk_type == "equal":
                syntax_chunks.append(Syntax("\n".join(chunk), syntax, theme="github-dark", line_numbers=True, start_line=line_number, background_color="rgb(0,0,0)"))
                line_number += len(chunk)
            elif chunk_type == "remove":
                syntax_chunks.append(Syntax(indent("\n".join(chunk), " "*4), syntax, theme="github-dark", line_numbers=False, background_color="rgb(80,0,0)"))
                num_lines_removed_so_far += len(chunk)
            else:
                raise ValueError(f"Unexpected chunk type: {chunk_type}")
        panel_title = f"Lines {start + 1} to {line_number}"
        panels.append(
            Panel(
                Group(*syntax_chunks),
                title=panel_title,
                border_style="bright_black",
            )
        )
    return Group(*panels)


class DiffStats(NamedTuple):
    """
    A named tuple containing the statistics of a diff.
    """

    added: int
    removed: int


def get_diff_statistics(difflines: list[str]) -> DiffStats:
    """
    Get the statistics of a diff.

    :param diff: The diff to get the statistics of.
    :return: A named tuple containing the statistics of the diff.
    """
    added = 0
    removed = 0
    for line in difflines:
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return DiffStats(added=added, removed=removed)
