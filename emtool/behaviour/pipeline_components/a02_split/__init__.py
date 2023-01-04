from emtool.structures.text_file import TextFile, TextFileFragment


def split_file(file: TextFile, num_lines: int = 10) -> list[TextFileFragment]:
    # Split the file into fragments
    contents = file.contents.splitlines()
    return [
        TextFileFragment(
            path=file.path,
            contents="\n".join(contents[i : i + num_lines]),
            start_line=i,
        )
        for i in range(0, len(contents), num_lines)
    ]
