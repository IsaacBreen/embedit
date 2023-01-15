from typing import Optional

import openai
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential

enc = tiktoken.get_encoding("gpt2")


def complete(
    string: str,
    prompt: str,
    pre_prompt: str,
    engine: str = "text-davinci-003",
    verbose: bool = False,
) -> str:
    """
    Takes a markdown string and a prompt, sends the prompt to the OpenAI API with the markdown string as the context,
    and returns the result.
    """
    end_prompt_token = "<| END OF PROMPT |>"
    pre_prompt_extra = f"Once you're finished responding, write {end_prompt_token} on a line by itself."
    total_prompt = "\n".join(
        [
            pre_prompt,
            pre_prompt_extra,
            "## Request",
            prompt,
            "## Input",
            string,
            "## Response",
            "<| START OF RESPONSE |>",
        ]
    )
    num_input_tokens = len(enc.encode(total_prompt))
    max_tokens = 4097 if engine != "code-davinci-002" else 8000
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
        stop=[end_prompt_token],
    )
    text = result.choices[0].text.strip()
    if verbose:
        print(f"Response:")
        print(text)
    return text


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-embedding-ada-002") -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: list[str], engine="text-embedding-ada-002"
) -> list[list[float]]:
    assert len(list_of_text) > 0
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.Embedding.create(input=list_of_text, engine=engine).data
    data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
    return [d["embedding"] for d in data]
