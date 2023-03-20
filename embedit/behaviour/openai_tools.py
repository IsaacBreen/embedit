import logging
from dataclasses import dataclass
from typing import Literal
from typing import Optional

import openai
import tiktoken
from delegatefn import delegate
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_random_exponential
from tqdm.auto import tqdm

from embedit.structures.special_tokens import end_response_token
from embedit.structures.special_tokens import start_response_token
from embedit.utils.log import logger


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)


enc = tiktoken.get_encoding("gpt2")


def toklen(string: str) -> int:
    """
    Returns the number of tokens in the given string.
    """
    return len(enc.encode(string))


def tokclip(string: str, max_tokens: int, keep: Literal["left", "right"]) -> str:
    """
    Returns the given string clipped to the given number of tokens.
    """
    if toklen(string) <= max_tokens:
        return string

    if keep == "left":
        return enc.decode(enc.encode(string)[-max_tokens:])
    elif keep == "right":
        return enc.decode(enc.encode(string)[:max_tokens])
    else:
        raise ValueError(f"Invalid value for keep: {keep}")


def get_max_tokens(model: str) -> int:
    """
    Returns the maximum number of tokens that can be sent to the given model.
    """
    # TODO: Not very robust. Should probably use the API to get this information.
    if model == "gpt-3.5-turbo":
        return 8000
    else:
        return 4000


def response_did_finished(response: str) -> bool:
    """
    Returns True if the given response contains the token that indicates that the response is finished.
    """
    return end_response_token in response


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
@delegate(openai.Completion.create)
def openai_create_raw(**kwargs) -> str:
    kwargs.setdefault("model", "gpt-3.5-turbo")
    kwargs["messages"] = [{"role": "system", "content": "You are a text completion engine."},
                          {"role": "user", "content": kwargs.pop("prompt")}]
    return openai.ChatCompletion.create(**kwargs).choices[0].message.content


@delegate(openai.Completion.create)
def openai_create(model: str, **kwargs) -> str:
    return openai_create_raw(model=model, **kwargs)


@dataclass(frozen=True)
class Task:
    context: str
    request: Optional[str] = None


@dataclass(frozen=True)
class Result:
    response: str


def complete(
    context: str,
    prompt: Optional[str],
    pre_prompt: str,
    *,
    examples: Optional[list[tuple[Task, Result]]] = None,
    model: str = "gpt-3.5-turbo",
    min_output_tokens: int = 1,
    max_output_tokens: Optional[int] = None,
    mark_examples: bool = False,
) -> str:
    """
    Takes a markdown string and a prompt, sends the prompt to the OpenAI API with the markdown string as the context,
    and returns the result.
    """
    if examples is None:
        examples = []

    example_parts = []
    for task, result in examples:
        if task.request is None:
            prompt_parts = []
        else:
            prompt_parts = [
                "## Request" + (" (Example)" if mark_examples else ""),
                task.request,
            ]
        example_parts.extend(
            [
                "## Context" + (" (Example)" if mark_examples else ""),
                task.context,
                *prompt_parts,
                "## Response" + (" (Example)" if mark_examples else ""),
                start_response_token,
                result.response,
                end_response_token,
            ]
        )

    if prompt is None:
        prompt_parts = []
    else:
        prompt_parts = [
            "## Request",
            prompt,
        ]

    pre_prompt_extra = f"Once you're finished responding, write {end_response_token} on a line by itself."
    total_prompt = "\n".join(
        [
            pre_prompt,
            pre_prompt_extra,
            *example_parts,
            "## Context",
            context,
            *prompt_parts,
            "## Response",
            start_response_token,
        ]
    )
    num_input_tokens = toklen(total_prompt)
    if max_output_tokens is None:
        max_tokens = get_max_tokens(model)
        max_output_tokens = max_tokens - num_input_tokens
    else:
        max_tokens = num_input_tokens + max_output_tokens
    if max_output_tokens < min_output_tokens:
        raise ValueError(
            f"max_tokens ({max_tokens}) is too small to fit the prompt ({num_input_tokens} tokens)."
        )
    request_params = dict(
        model=model,
        prompt=total_prompt,
        max_tokens=max_output_tokens,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=[end_response_token],
    )
    logger.info(f"Parameters: {request_params}")
    logger.info(f"Prompt: {total_prompt}")

    text = openai_create(**request_params)
    logger.info(f"Response (including end token): {text}")
    # If the response ran out of tokens, raise an exception
    if toklen(text) == max_output_tokens + 1:
        raise ValueError(
            "Ran out of tokens. Try setting max_tokens higher."
        )
    # Remove the end token
    text = text.replace(end_response_token, "")
    return text


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model="text-embedding-ada-002") -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def get_embeddings(
    list_of_text: list[str], model="text-embedding-ada-002", batch_size: Optional[int] = None
) -> list[list[float]]:
    assert len(list_of_text) > 0
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    create = retry(
        wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3)
    )(openai.Embedding.create)

    def _get_embeddings(list_of_text: list[str]) -> list[list[float]]:
        data = create(input=list_of_text, model=model).data
        data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
        return [d["embedding"] for d in data]

    if batch_size is None:
        # If the batch size is not specified, send all the data at once.
        return _get_embeddings(list_of_text)
    else:
        # Send in batches. Log to the logger with tqdm.
        embeddings = []
        logger.addHandler(TqdmLoggingHandler())

        # Only show the progress bar if the log level is INFO or lower.
        if logger.level >= logging.INFO:
            pbar = tqdm(total=len(list_of_text), desc="Getting embeddings")
        else:
            pbar = None

        for i in range(0, len(list_of_text), batch_size):
            batch = list_of_text[i: i + batch_size]
            embeddings.extend(_get_embeddings(batch))

            if pbar:
                pbar.update(batch_size)
                pbar.set_description(f"Processing {i + batch_size}/{len(list_of_text)} texts")
                pbar.refresh()

        if pbar:
            pbar.close()

    return embeddings
