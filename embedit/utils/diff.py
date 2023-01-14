import difflib
from typing import NamedTuple, Optional

from rich.console import Group, Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.style import Style
from rich.text import Text, Span
import textwrap

console = Console()

DEFAULT_INDENT_WIDTH = 4

def remove_background_color(text: Text) -> Text:
    """
    Remove the background color from a Text object.

    :param text: The Text object to remove the background color from.
    :return: The Text object with the background color removed.
    """
    # This is more complicated than it ought to be. `rich` makes it very difficult to modify its objects (lesson: if
    # you're going to freeze objects, at least provide an `evolve` or `replace` function).

    # Remove from spans
    spans = []
    for span in text.spans:
        if isinstance(span.style, Style):
            spans.append(Span(start=span.start, end=span.end, style=Style(color=span.style.color)))
        else:
            spans.append(span)
    # Remove from the main text
    return Text(text.plain, spans=spans)


def add_line_numbers(text: Text, *, start: int = 1, width: int = DEFAULT_INDENT_WIDTH) -> Text:
    """
    Add line numbers to a Text object.

    :param text: The Text object to add line numbers to.
    :param start: The line number to start with.
    :param width: The width of the line number column.
    :return: The Text object with line numbers added.
    """
    # Add line numbers to the text
    lines = text.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines, start=start):
        line_number = f"{i:>{width-1}} "
        numbered_lines.append(Text(line_number, style="dim") + line)
    # Join the lines
    return Text('\n').join(numbered_lines)


def indent(text: Text, prefix: str) -> Text:
    """
    Indent a Text object.

    :param text: The Text object to indent.
    :param prefix: The prefix to indent the Text object with.
    :return: The indented Text object.
    """
    return Text(textwrap.indent(text.plain, prefix), spans=text.spans, style=text.style)


def pretty_diff(a: str, b: str, context: Optional[int] = 3, syntax: str = "python"):
    """
    Pretty print a diff.

    :param a: The first string to diff.
    :param b: The second string to diff.
    :param context: The number of lines of context to show around each change.
    :param syntax: The syntax to highlight the diff with.
    :return: A pretty printed diff.
    """
    # To get proper syntax highlighting, we need to render each string separately, combine them, and split into the
    # intervals above. Render the texts separately with syntax highlighting.
    theme = "github-dark"
    a_with_syntax = Syntax(a, syntax, theme=theme)
    b_with_syntax = Syntax(b, syntax, theme=theme)
    # Just get the highlighted text
    a_pretty_lines = a_with_syntax.highlight(a_with_syntax.code)
    b_pretty_lines = b_with_syntax.highlight(b_with_syntax.code)
    # Add line numbers to b (which we lost when we highlighted the text 🤦🏻‍). Indent a so that it lines up with b.
    a_pretty_lines = indent(a_pretty_lines, " " * DEFAULT_INDENT_WIDTH)
    b_pretty_lines = add_line_numbers(b_pretty_lines)
    # Remove the background color from the texts
    a_pretty_lines = remove_background_color(a_pretty_lines)
    b_pretty_lines = remove_background_color(b_pretty_lines)
    # Split into lines
    a_pretty_lines = a_pretty_lines.split("\n")
    b_pretty_lines = b_pretty_lines.split("\n")
    # Splice the strings together
    a_pretty_lines_iter = iter(a_pretty_lines)
    b_pretty_lines_iter = iter(b_pretty_lines)
    diff = []
    difflines = list(difflib.Differ().compare(a.splitlines(), b.splitlines()))
    for line in difflines:
        try:
            if line.startswith("+"):
                b_line = next(b_pretty_lines_iter)
                diff.append(Text(b_line.plain, spans=b_line.spans, style=Style(bgcolor="rgb(0,90,0)"), justify="left"))
            elif line.startswith("-"):
                a_line = next(a_pretty_lines_iter)
                diff.append(Text(a_line.plain, spans=a_line.spans, style=Style(bgcolor="rgb(90,0,0)"), justify="left"))
            elif line.startswith("  "):
                next(a_pretty_lines_iter)
                b_line = next(b_pretty_lines_iter)
                diff.append(Text(b_line.plain, spans=b_line.spans))
        except StopIteration:
            pass
    if context is None:
        return Group(*diff)
    else:
        # Compute the in-context lines
        intervals = []
        for i, line in enumerate(difflines):
            if line.startswith("+") or line.startswith("-"):
                if intervals and intervals[-1][1] >= i - context:
                    # Extend the previous interval
                    intervals[-1] = (intervals[-1][0], min(i + context, len(diff)))
                else:
                    # Start a new interval
                    intervals.append((max(0, i - context), min(i + context, len(diff))))
        # Split the diff into chunks and render them as panels
        diff_chunks = [
            Panel(Group(*diff[start:end+1]), title=f"{start+1}-{end}")
            for start, end in intervals
        ]
        return Group(*diff_chunks)


class DiffStats(NamedTuple):
    """
    A named tuple containing the statistics of a diff.
    """

    added: int
    removed: int


def get_diff_stats(a: str, b: str) -> DiffStats:
    """
    Get the statistics of a diff.

    :param a: The first string to diff.
    :param b: The second string to diff.
    :return: A named tuple containing the statistics of the diff.
    """
    difflines = list(difflib.Differ().compare(a.splitlines(), b.splitlines()))
    added = 0
    removed = 0
    for line in difflines:
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return DiffStats(added=added, removed=removed)
