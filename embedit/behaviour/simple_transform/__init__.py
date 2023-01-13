from typing import Optional

import openai
import tiktoken
from dir2md import dir2md, md2dir, save_dir

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
    overwrite: bool = False, engine: str = "text-davinci-003", verbose: bool = False
):
    """
    Transform the given files by passing their markdown representation with the given prompt to the OpenAI API.
    """
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
                    *chunk_files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, overwrite=overwrite,
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
            *chunk_files, prompt=prompt, pre_prompt=pre_prompt, output_dir=output_dir, overwrite=overwrite,
            engine=engine, verbose=verbose
        )
    else:
        # Transform all files at once
        markdown = "\n".join(dir2md(*files))
        results = simple_transform_markdown_dir(
            markdown, prompt=prompt, pre_prompt=pre_prompt, engine=engine, verbose=verbose
        )
        save_dir(list(md2dir(results)), output_dir=output_dir)
        return results


def simple_transform_markdown_dir(
    string: str, prompt: str, pre_prompt: Optional[str] = None, engine: str = "text-davinci-003", verbose: bool = False
) -> str:
    """
    Takes a markdown string and a prompt, sends the prompt to the OpenAI API with the markdown string as the context,
    and returns the result.
    """
    if pre_prompt is None:
        pre_prompt = default_pre_prompt
    total_prompt = "\n".join([pre_prompt, "## Request", prompt, "## Input", string, "## Response"])
    num_input_tokens = len(enc.encode(total_prompt))
    max_tokens = 4097
    max_output_tokens = max_tokens - num_input_tokens
    if verbose:
        print(f"Prompt:")
        print(total_prompt)
    result = openai.Completion.create(
        engine=engine,
        prompt=total_prompt,
        max_tokens=max_output_tokens,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    text = result.choices[0].text.strip()
    if verbose:
        print(f"Response:")
        print(text)
    return text
