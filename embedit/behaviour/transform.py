from typing import Optional

import tiktoken
from dir2md import dir2md, md2dir, save_dir

from embedit.behaviour.openai import complete

enc = tiktoken.get_encoding("gpt2")

default_pre_prompt = " ".join(
    [
        "You are an advanced AI assistant.",
        "Respond to the user's requests with the appropriate text.",
        "Write in a professional manner and maintain the highest possible standards of output.",
        "If the user's input is has a particular format (e.g. markdown), the response shall be faithful to that format.",
        "For example, if the user's input is a series of markdown fences prepended by a filename in a comment, the response shall be a series of markdown fences prepended by the same filename in a comment.",
    ]
)


def simple_transform_files(
    *files, prompt: str, pre_prompt: Optional[str] = None, output_dir: str, max_chunk_len: Optional[int] = None,
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
            *chunk_files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, yes=yes,
            engine=engine, verbose=verbose
        )
    else:
        # Transform all files at once
        markdown = "\n".join(dir2md(*files))
        results = complete(
            markdown, prompt=prompt, pre_prompt=pre_prompt, engine=engine, verbose=verbose
        )
        save_dir(list(md2dir(results)), output_dir=output_dir, yes=yes)
        return results
