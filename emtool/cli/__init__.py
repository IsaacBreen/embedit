import pathlib
from typing import Literal

import fire
from delegatefn import delegate
from rich.console import Console
from rich.syntax import Syntax

from emtool.structures.embedding import EmbeddedTextFileFragmentSimilarityResult

console = Console()

from emtool.behaviour.pipelines import semantic_search


def center_pad(text: str, width: int, *, fillchar: str = " ") -> str:
    text = f" {text.strip()} "
    # Calculate the number of characters to pad on each side
    padding = (width - len(text)) // 2
    # Pad the text
    return fillchar * padding + text + fillchar * padding


@delegate(semantic_search, ignore=["query", "files"])
def search(query: str, *files: str, order: Literal["ascending", "descending"] = "ascending", **kwargs):
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
        # console.print(center_pad("End of result", width=80, fillchar="-"), style="bold")
        console.print()


def main():
    fire.Fire(search)


if __name__ == "__main__":
    main()
