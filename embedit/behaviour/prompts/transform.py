from embedit.behaviour.create import default_pre_prompt
from embedit.structures.special_tokens import end_response_token
from embedit.structures.special_tokens import start_response_token

default_transform_pre_prompt = default_pre_prompt

default_transform_pre_prompt = " ".join(
    [
        default_transform_pre_prompt,
        "For example, if the user's input is a series of markdown fences prepended by a filename in a comment,",
        "the response shall be a series of markdown fences prepended by the same filename in a comment.",
    ]
)

default_transform_pre_prompt = "\n".join(
    [
        default_transform_pre_prompt,
        "If you need to output files, they should be a sequence of markdown fences, each preceded by a filename in a comment.",
        "<| BEGINNING OF EXAMPLE |>",
        start_response_token,
        "<!-- relative/path/to/file.py -->" "```python",
        "def hello_world():",
        '    print("Hello, world!")',
        "```",
        "<!-- relative/path/to/another/file.py -->",
        "```python",
        "def another_function():",
        "    ...",
        "```",
        end_response_token,
        "<| END OF EXAMPLE |>",
    ]
)
