from typing import Optional

from dir2md import md2dir, save_dir

from embedit.behaviour.openai import complete

default_pre_prompt = " ".join(
    [
        "You are an advanced AI assistant for creating files.",
        "Respond to the user's requests with the appropriate text.",
        "Write in a professional manner and maintain the highest possible standards of output.",
        "Your response should be a sequence of markdown fences, each preceded by a filename in a comment.",
    ]
)

default_pre_prompt = "\n".join(
    [
        default_pre_prompt,
        "<| BEGINNING OF EXAMPLE |>",
        "<!-- relative/path/to/file.py -->"
        "```python",
        "def hello_world():",
        "    print(\"Hello, world!\")",
        "```",
        "<!-- relative/path/to/another/file.py -->",
        "```python",
        "def another_function():",
        "    ...",
        "```",
        "<| END OF EXAMPLE |>",
    ]
)


def create(
    prompt: str, *, pre_prompt: Optional[str] = None, output_dir: str = "out",
    yes: bool = False, engine: str = "text-davinci-003", verbose: bool = False
):
    """
    Create files from a prompt and pre-prompt using OpenAI's GPT-3 engine.

    Args:
        prompt (str): The prompt to be used for generating the files.
        pre_prompt (Optional[str], optional): The pre-prompt to be used for generating the files. Defaults to None.
        output_dir (str, optional): The directory to save the generated files to. Defaults to "out".
        yes (bool, optional): Whether to overwrite existing files. Defaults to False.
        engine (str, optional): The OpenAI engine to use. Defaults to "text-davinci-003".
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        list: A list of markdown fences, each preceded by a filename in a comment.
    """
    if pre_prompt is None:
        pre_prompt = default_pre_prompt

    results = complete(
        "<| No input |>", prompt=prompt, pre_prompt=pre_prompt, engine=engine, verbose=verbose
    )
    save_dir(list(md2dir(results)), output_dir=output_dir, yes=yes)
    return results