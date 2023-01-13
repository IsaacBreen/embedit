from embedit.structures.text_file import TextFile, TextFileFragment


def combine_fragments(fragments: list[TextFileFragment]) -> TextFile:
    # Combine the fragments into a single file
    # Order the fragments by their start line
    fragments.sort(key=lambda fragment: fragment.start_line)
    # Combine the fragments into a single file
    return TextFile(path=fragments[0].path, contents="".join(fragment.contents for fragment in fragments))
