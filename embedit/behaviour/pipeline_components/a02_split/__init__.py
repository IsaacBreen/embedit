from embedit.structures.text_file import TextFile, TextFileFragment


def split_file(file: TextFile, *, fragment_lines: int = 10, ignore_empty: bool = True) -> list[TextFileFragment]:
    # Split the file into fragments
    contents = file.contents.splitlines()
    fragments = [
        TextFileFragment(
            path=file.path,
            contents="\n".join(contents[i: i + fragment_lines]),
            start_line=i,
        )
        for i in range(0, len(contents), fragment_lines)
    ]
    if ignore_empty:
        fragments = [fragment for fragment in fragments if fragment.contents]
    return fragments