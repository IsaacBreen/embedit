import pathlib
from typing import Literal, Optional

import fire
from delegatefn import delegate
from rich.console import Console
from rich.syntax import Syntax

from embedit.behaviour.create import create
from embedit.behaviour.transform import simple_transform_files
from embedit.structures.embedding import EmbeddedTextFileFragmentSimilarityResult

console = Console()

from embedit.behaviour.search.pipelines import semantic_search


def center_pad(text: str, width: int, *, fillchar: str = " ") -> str:
    text = f" {text.strip()} "
    # Calculate the number of characters to pad on each side
    padding = (width - len(text)) // 2
    # Pad the text
    return fillchar * padding + text + fillchar * padding


@delegate(semantic_search, ignore={"query", "files"})
def search(query: str, *files: str, order: Literal["ascending", "descending"] = "ascending", **kwargs):
    """a command line tool for semantic file search

    `embedit search` is a command line tool for performing semantic searches on a set of text files. It allows you to specify a search query and a list of text files to search, and returns a list of results ranked by their similarity to the query.

    :param query: The search query string.
    :param files: One or more text files to search.
    :param order: The order in which to sort the search results. Can be 'ascending' or 'descending'.
    :param verbosity: An integer indicating the level of verbosity of the search results. Higher numbers will produce more detailed output.
    :param fragment_lines: The number of lines to include in each search result fragment.
    :param min_fragment_lines: The minimum number of lines that must match the search query for a result to be included.
    :param threshold: A float indicating the minimum similarity score a result must have to be included.
    :param top_n: An integer indicating the maximum number of search results to return.
    :return: A list of search results, ranked by their similarity to the query.
    :raises: ValueError - If the ``--order`` argument is not 'ascending' or 'descending'.
    """
    directories = [file for file in files if pathlib.Path(file).is_dir()]
    if directories:
        console.print(f"Ignoring directories: {', '.join(directories)}")
        files = [file for file in files if not pathlib.Path(file).is_dir()]
    assert len(files) > 0, "No files were provided"
    # Filter out directories
    console.print(f"Searching for '{query}' in {len(files)} files")
    # Search for the query
    results: list[EmbeddedTextFileFragmentSimilarityResult] = semantic_search(query, *files, **kwargs)
    # Enumerate and sort the results
    results = sorted(results, key=lambda result: result.similarity, reverse=True)
    enumerated_results = enumerate(results, start=1)
    if order == "ascending":
        enumerated_results = reversed(list(enumerated_results))
    # Print the results
    console.print(f"Found {len(results)} results")
    for i, result in enumerated_results:
        header = f"Result {i}"
        # Pad the result header with hyphens
        header = center_pad(header, width=80, fillchar="-")
        print(header)
        # Print result info
        console.print(f"Similarity: {result.similarity:.2f}")
        console.print(f"Path: {result.embedded_fragment.fragment.path}")
        # Print the result contents with appropriate highlighting
        lexer: str = Syntax.guess_lexer(
            result.embedded_fragment.fragment.path, result.embedded_fragment.fragment.contents
        )
        console.print(
            Syntax(
                result.embedded_fragment.fragment.contents, lexer, theme="monokai", line_numbers=True,
                start_line=result.embedded_fragment.fragment.start_line
            )
        )
        console.print()


def transform(
    *files, prompt: str, pre_prompt: Optional[str] = None, output_dir: str = "out", max_chunk_len: Optional[int] = None,
    yes: bool = None, engine: str = "code-davinci-002", verbose: bool = False
):
    """
    Transforms text files by passing their markdown representation to the OpenAI API.
    :param files: Text files to transform.
    :param prompt: The prompt to pass to the OpenAI API.
    :param pre_prompt: An optional pre-prompt to pass to the OpenAI API.
    :param output_dir: Directory to save the transformed files.
    :param max_chunk_len: Maximum length of chunks to pass to the OpenAI API.
    :param yes: Whether to prompt before creating or overwriting files.
    :param engine: The OpenAI API engine to use. Defaults to 'code-davinci-002'; however, if you have access to "code-davinci-002", I recommend using that instead.
    :param verbose: Print verbose output.
    :return: Output of the OpenAI API.
    """
    simple_transform_files(
        *files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, max_chunk_len=max_chunk_len, yes=yes,
        engine=engine, verbose=verbose
    )


def main():
    fire.Fire({"search": search, "transform": transform, "create": create})


if __name__ == "__main__":
    main()
