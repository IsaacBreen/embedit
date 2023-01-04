import pathlib
from emtool.structures.text_file import TextFile, TextFileFragment

def gather(*files: str) -> list[TextFile]:
    # Gather the files into a list of TextFile objects
    return [TextFile(path=pathlib.Path(file), contents=pathlib.Path(file).read_text()) for file in files]
