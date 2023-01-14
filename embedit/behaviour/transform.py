import pathlib
from typing import Optional

import tiktoken
from dir2md import TextFile, dir2md, md2dir, save_dir
from rich import print
from rich.panel import Panel
from rich.syntax import Syntax

from embedit.behaviour.openai import complete
from embedit.utils.diff import diff_strings, get_diff_statistics, prettify_diff

enc = tiktoken.get_encoding("gpt2")

default_pre_prompt = " ".join(
    [
        "You are an advanced AI assistant.",
        "Respond to the user's requests with the appropriate text.",
        "Write in a professional manner and maintain the highest possible standards of output.",
        "If the user's input is has a particular format (e.g. markdown), the response shall be faithful to that format.",
        "For example, if the user's input is a series of markdown fences prepended by a filename in a comment,",
        "the response shall be a series of markdown fences prepended by the same filename in a comment.",
    ]
)

default_pre_prompt = "\n".join(
    [
        default_pre_prompt,
        "If you need to output files, they should be a sequence of markdown fences, each preceded by a filename in a comment.",
        "<| BEGINNING OF EXAMPLE |>",
        "<!-- relative/path/to/file.py -->"
        "```python",
        "def hello_world():",
        "    print(\"Hello, world!\")",
        "```",
        "<!-- relative/path/to/another/file.py -->",
        "```python",
        "def another_function():",
        "    ...",
        "```",
        "<| END OF EXAMPLE |>",
    ]
)


def get_difflines(result: str, path: str) -> list[str]:
    """
    Compare the result of a transformation to the original file and return a pretty diff.
    """
    # Read the original file if it exists.
    original = pathlib.Path(path).read_text() if pathlib.Path(path).is_file() else ""
    # Diff the result and the original
    difflines: list[str] = diff_strings(original, result)
    return difflines


def simple_transform_files(
    *files, prompt: str, pre_prompt: Optional[str] = None, output_dir: str,
    max_chunk_len: Optional[int] = int(3.5 * 1600),
    yes: bool = False, engine: str = "text-davinci-003", verbose: bool = False
):
    """
    Transform the given files by passing their markdown representation with the given prompt to the OpenAI API.
    """
    if pre_prompt is None:
        pre_prompt = default_pre_prompt

    if max_chunk_len is not None:
        # Add as many files as possible while keeping the total length of the markdown representation below
        # max_chunk_len
        chunk_strings = []
        chunk_files = []
        for file in files:
            file_string = "\n".join(dir2md(file))
            if sum(len(string) for string in chunk_strings) + len(file_string) > max_chunk_len:
                # The current chunk is too long, recurse with the current chunk
                simple_transform_files(
                    *chunk_files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, yes=yes,
                    engine=engine, verbose=verbose
                )
                # Start a new chunk
                chunk_strings = []
                chunk_files = []
            # Add the file to the current chunk
            chunk_strings.append(file_string)
            chunk_files.append(file)
        # Recurse with the last chunk
        return simple_transform_files(
            *chunk_files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, yes=yes, engine=engine,
            verbose=verbose
        )
    else:
        # Transform all files at once
        markdown = "\n".join(dir2md(*files))
        result_markdown: str = complete(
            string=markdown, prompt=prompt, pre_prompt=pre_prompt, engine=engine, verbose=verbose
        )
        result_files: list[TextFile] = list(md2dir(result_markdown))
        # Print the diff of each file
        for result in result_files:
            print(prettify_file_diff(result.text, result.path))
        save_dir(result_files, output_dir=output_dir, yes=yes)
        return result_files


def prettify_file_diff(result: str, path: str):
    """
    Compare the result of a transformation to the original file and return a pretty diff.
    """
    # Read the original file if it exists.
    original = pathlib.Path(path).read_text() if pathlib.Path(path).is_file() else ""
    # Diff the result and the original
    difflines: list[str] = diff_strings(original, result)
    diff_stats = get_diff_statistics(difflines)
    # Pretty print the diff in a panel
    if pathlib.Path(path).is_file():
        title = f"Diff for {path}:"
        subtitle = f"Lines added: {diff_stats.added}; lines removed: {diff_stats.removed}"
    else:
        title = f"Lines added: {diff_stats.added}"
        subtitle = f"New file {path}:"
    pretty_diff = prettify_diff(difflines, syntax=Syntax.guess_lexer(path, result))
    return Panel(pretty_diff, title=title, subtitle=subtitle)
