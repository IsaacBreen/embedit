import pathlib

from embedit.structures.text_file import TextFile


def gather(*files: str, ignore_empty: bool = True) -> list[TextFile]:
    # Gather the files into a list of TextFile objects
    results = [TextFile(path=pathlib.Path(file), contents=pathlib.Path(file).read_text()) for file in files]
    if ignore_empty:
        results = [result for result in results if result.contents]
    return results
