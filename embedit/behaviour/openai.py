import openai
import tiktoken
from rich import print
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential

enc = tiktoken.get_encoding("gpt2")


def toklen(string: str) -> int:
    """
    Returns the number of tokens in the given string.
    """
    return len(enc.encode(string))


def get_max_tokens(engine: str) -> int:
    """
    Returns the maximum number of tokens that can be sent to the given engine.
    """
    # TODO: Not very robust. Should probably use the API to get this information.
    if engine == "code-davinci-002":
        return 8000
    else:
        return 4000


def response_did_finished(response: str) -> bool:
    """
    Returns True if the given response contains the token that indicates that the response is finished.
    """
    return "<| END OF PROMPT |>" in response


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def openai_create_raw(**kwargs):
    return openai.Completion.create(**kwargs)

def openai_create(**kwargs) -> str:
    if kwargs["engine"] == "code-davinci-002":
        return openai_create_codex(**kwargs)
    else:
        return openai_create_raw(**kwargs).choices[0].text


def openai_create_codex(**kwargs) -> str:
    """
    Codex has a rate limit of 2000 tokens per second, excluding the prompt, which means we need to call it multiple
    times if we want a completion longer than 2000 tokens.
    """
    response = ""
    max_tokens = kwargs["max_tokens"]
    while not response_did_finished(response) and max_tokens > 0:
        kwargs["max_tokens"] = min(max_tokens, 2000)
        response += openai_create_raw(**kwargs).choices[0].text
        kwargs["prompt"] += response
        max_tokens -= 2000
    return response


def complete(
    string: str,
    prompt: str,
    pre_prompt: str,
    engine: str = "code-davinci-002",
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
    num_input_tokens = toklen(total_prompt)
    max_tokens = get_max_tokens(engine)
    max_output_tokens = max_tokens - num_input_tokens
    request_params = dict(
        engine=engine,
        prompt=total_prompt,
        max_tokens=max_output_tokens,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[end_prompt_token]
    )
    if verbose:
        print("Parameters:")
        print(request_params)
        print(f"Prompt:")
        print(total_prompt)

    text = openai_create(**request_params)
    if verbose:
        print(f"Response:")
        print(text)
    # If the response ran out of tokens, raise an exception
    if toklen(text) == max_output_tokens + 1:
        raise ValueError(
            "Ran out of tokens. Try setting max_tokens higher. (text-davinci-003 supports up to 4097, code-davinci-002 supports up to 8000)"
        )
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

    create = retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))(openai.Embedding.create)
    data = create(input=list_of_text, engine=engine).data
    data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
    return [d["embedding"] for d in data]
