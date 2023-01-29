import pathlib
from typing import Optional
from typing import Sequence

import tiktoken
from dir2md import TextFile
from dir2md import dir2md
from dir2md import md2dir
from dir2md import save_dir
from rich import print
from rich.panel import Panel

from embedit.behaviour.openai import complete
from embedit.behaviour.openai import toklen
from embedit.utils.diff import get_diff_stats
from embedit.utils.diff import pretty_diff

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
        "<!-- relative/path/to/file.py -->" "```python",
        "def hello_world():",
        '    print("Hello, world!")',
        "```",
        "<!-- relative/path/to/another/file.py -->",
        "```python",
        "def another_function():",
        "    ...",
        "```",
        "<| END OF EXAMPLE |>",
    ]
)

def simple_transform_files(
    *files,
    prompt: str,
    pre_prompt: Optional[str] = None,
    output_dir: str,
    max_chunk_len: Optional[int] = 1600,
    yes: bool = False,
    engine: str = "code-davinci-002",
    verbose: bool = False,
):
    """
    Transform the given files by passing their markdown representation with the given prompt to the OpenAI API.
    """
    if pre_prompt is None:
        pre_prompt = default_pre_prompt

    if max_chunk_len is None:
        chunks = [files]
    else:
        chunks = simple_transform_files_get_chunks(
            engine, files, max_chunk_len, output_dir, pre_prompt, prompt, verbose, yes
        )

    results = []
    for chunk_files in chunks:
        result = simple_transform_files_execute_chunk(engine, chunk_files, pre_prompt, prompt, verbose)
        results.extend(result)

    wrapup(results, output_dir, yes)





def simple_transform_files_get_chunks(
    engine: str,
    files: Sequence[str],
    max_chunk_len: int,
    output_dir: str,
    pre_prompt: str,
    prompt: str,
    verbose: bool,
    yes: bool,
):
    # Add as many files as possible while keeping the total length of the markdown representation below
    # max_chunk_len tokens
    chunk_strings = []
    chunk_files = []
    for file in files:
        file_string = "\n".join(dir2md(file))
        if (
                sum(toklen(string) for string in chunk_strings) + toklen(file_string)
                > max_chunk_len
        ):
            # The current chunk plus the new file would exceed max_chunk_len, so transform the current chunk
            yield chunk_files
            # Start a new chunk
            chunk_strings = []
            chunk_files = []
        # Add the file to the current chunk
        chunk_strings.append(file_string)
        chunk_files.append(file)
    # Recurse with the last chunk
    yield chunk_files

def simple_transform_files_execute_chunk(
    engine: str,
    files: Sequence[TextFile],
    pre_prompt: str,
    prompt: str,
    verbose: bool
) -> list[TextFile]:
    # Transform all files at once
    markdown = "\n".join(dir2md(*files))
    result_markdown: str = complete(
        string=markdown,
        prompt=prompt,
        pre_prompt=pre_prompt,
        engine=engine,
        verbose=verbose,
    )
    result_files: list[TextFile] = list(md2dir(result_markdown))
    return result_files

def wrapup(result_files: list[TextFile], output_dir: str, yes: bool):
    # Print the diff of each file
    for result in result_files:
        original = (
            pathlib.Path(result.path).read_text()
            if pathlib.Path(result.path).is_file()
            else ""
        )
        diff = pretty_diff(original, result.text)
        diff_stats = get_diff_stats(original, result.text)
        print(
            Panel(
                diff,
                title=result.path,
                subtitle=f"{diff_stats.added} lines added, {diff_stats.removed} lines removed",
            )
        )
    save_dir(result_files, output_dir=output_dir, yes=yes)
