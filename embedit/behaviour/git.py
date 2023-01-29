"""
Create a git commit message.
"""
import os
import re
from typing import Iterator
from typing import Optional

from git import Repo

from embedit.behaviour.openai_tools import Result
from embedit.behaviour.openai_tools import Task
from embedit.behaviour.openai_tools import complete
from embedit.behaviour.openai_tools import tokclip
from embedit.behaviour.openai_tools import toklen
from embedit.behaviour.prompts.default import default_pre_prompt

musings_on_good_vs_great_commit_messages = """
A good commit message should:

- Be written in the present tense
- Be brief and to the point (less than 50 characters is a good rule of thumb for the subject line)
- Describe the changes made in the commit
A great commit message goes beyond that and also:

- Explains the why behind the change, not just the what
- Provides context for the change
- Is written in a clear and concise manner
- Includes any relevant issue or ticket numbers
- Includes any testing done

Writing clear and informative commit messages makes it easier for others to understand the changes made to the codebase and can help when debugging and reviewing pull requests.

There is no need to use a period to end very short commit messages, but longer commit messages should be written with formal grammar and punctuation.

Good commit message:

"Fix bug in login form"
Great commit message:

"Fix incorrect validation in login form #1234
The form was failing to validate the password field correctly and throwing an error, now the validation has been updated to correctly match the password against the database. Tested on multiple scenarios and confirmed to work as expected."

The "good" commit message is brief and clearly states the changes made, but it does not explain why the change was necessary or provide any context.

The "great" commit message, on the other hand, not only states the change but also explains the reasoning behind it, provides context, and includes relevant information such as the issue number and testing done. This level of detail makes it easier for others to understand the change and its impact on the codebase.
"""

pre_prompt_commit = "\n\n".join(
    [
        default_pre_prompt,
        musings_on_good_vs_great_commit_messages,
    ]
)


def make_builtin_examples():
    return [
        (
            Task(
                context="\n".join(
                    [
                        "diff --git a/file1.py b/file1.py",
                        "index 1b0b0b0..2b2b2b2 100644",
                        "--- a/file1.py",
                        "+++ b/file1.py",
                        "@@ -1,5 +1,7 @@",
                        " def func1():",
                        "     x = 5",
                        "-    print(x)",
                        "+    print(x + 2)",
                        '+    print("The value of x is: " + str(x))',
                        "     return x",
                    ]
                ),
                request=make_prompt(quality=10),
            ),
            Result(response="Modified func1 to print x + 2"),
        ),
        (
            Task(
                context="\n".join(
                    [
                        "diff --git a/file1.py b/file1.py",
                        "index 1b0b0b0..2b2b2b2 100644",
                        "--- a/file1.py",
                        "+++ b/file1.py",
                        "@@ -1,5 +1,7 @@",
                        " def func1():",
                        "     x = 5",
                        "-    print(x)",
                        "+    print(x + 2)",
                        '+    print("The value of x is: " + str(x))',
                        "     return x",
                    ]
                ),
                request=make_prompt(quality=10),
            ),
            Result(
                response="Modified func1 to print x + 2 and added a print statement to show the current value of x. This will help with debugging and understanding the function's behavior."
            ),
        ),
    ]


def make_prompt(quality: int, hint: Optional[str] = None) -> str:
    prompt = f"Write a commit message for the given diff (quality: {quality}/10)"
    if hint is not None:
        prompt += f" (hint: {hint})"
    return prompt


def rate_commit_message(commit_message: str) -> int:
    """
    Rate the given commit message on a scale of 0 to 10.
    """
    pre_prompt = "\n\n".join(
        [
            "Comment on the following commit messages. Rate them on a scale of 0 to 10.",
            musings_on_good_vs_great_commit_messages,
        ]
    )

    examples = [
        (
            Task(
                context="added function to add two numbers",
            ),
            Result(
                response="This commit message is too vague. It doesn't describe what the function does. It is repetitive, using the word 'add' twice. It lacks correct grammar. It should be 'Added a function to add two numbers. (RATING: 1/10)"
            ),
        ),
        (
            Task(
                context="Modified func1 to print x + 2",
            ),
            Result(
                response="This commit message describes the change but doesn't provide any context or reasoning behind the change. It is be written in the present tense, which is good. Be brief and to the point (less than 50 characters is a good rule of thumb for the subject line). (RATING: 4/10)"
            ),
        ),
        (
            Task(
                context="Modified func1 to print x + 2 and added a print statement to show the current value of x. This will help with debugging and understanding the function's behavior."
            ),
            Result(
                response="This commit message describes the change, the reasoning behind it and provides context. It is written in a clear and concise manner. This will be very helpful when debugging and understanding the function's behavior. (RATING: 7/10)"
            ),
        ),
    ]

    comment_and_rating = complete(
        context=commit_message,
        prompt=None,
        pre_prompt=pre_prompt,
        examples=examples,
        engine="text-davinci-003",
    )

    rating = re.search(r"\(RATING: (\d)/10\)", comment_and_rating).group(1)
    return int(rating)


def get_examples(
    num_examples, *, path: str = ".", max_log_tokens: int
) -> list[tuple[Task, Result]]:
    repo = Repo(path)
    token_count = 0
    examples = []
    # Starting from the most recent commit, add the commit message and diff to the examples until we reach the
    # max_log_tokens limit
    for commit in repo.iter_commits():
        if len(examples) >= num_examples:
            break
        message = commit.message
        diff = commit.diff(
            commit.parents[0] if commit.parents else None, create_patch=True
        )
        diff_str = "\n".join(str(d) for d in diff)
        this_token_count = toklen(message) + toklen(diff_str) + 20
        if token_count + this_token_count > max_log_tokens:
            continue
        token_count += this_token_count
        message_quality = rate_commit_message(message)
        task = Task(context=diff_str, request=make_prompt(quality=message_quality))
        result = Result(response=message)
        examples.append((task, result))
    return examples


def get_diffstrs(path: str, max_diff_tokens: int) -> Iterator[str]:
    """
    Yield diff strings for staged files in the given paths.

    If all the diffs together are within the max_diff_tokens limit, yield them all together. Otherwise, yield them one at a time.
    """
    repo = Repo(path)
    # Diff between head and staged
    diff = list(repo.index.diff(repo.head.commit, create_patch=True, R=True))
    if len(diff) == 0:
        raise ValueError("No changes to commit. Have you staged your changes?")
    diff_chunk = []
    diff_chunk_size = 0
    for d in diff:
        this_diff_size = toklen(str(d))
        if this_diff_size > max_diff_tokens:
            # If a single diff is too large, truncate it and yield it separately
            yield tokclip(str(d), max_diff_tokens, keep="right")
            continue
        diff_chunk_size += this_diff_size
        if diff_chunk_size > max_diff_tokens:
            yield "\n".join(diff_chunk)
            diff_chunk = []
            diff_chunk_size = 0
        diff_chunk.append(str(d))
    if len(diff_chunk) > 0:
        yield "\n".join(diff_chunk)


def make_commit_message(
    path: str = ".",
    max_log_tokens: int = 1400,
    max_diff_tokens: int = 1400,
    max_output_tokens: int = 400,
    engine: str = "code-davinci-002",
    num_examples: int = 10,
    use_builtin_examples: bool = True,
    hint: Optional[str] = None,
    num_lines_context: int = 10,
    verbose: bool = False,
) -> str:
    """Return a commit message based on the given diff."""
    # Set the GIT_DIFF_OPTS environment variable to change the number of lines of context shown in the diff.
    os.environ["GIT_DIFF_OPTS"] = f"-u{num_lines_context}"
    diffstrs = list(get_diffstrs(path=path, max_diff_tokens=max_diff_tokens))
    if use_builtin_examples:
        examples = make_builtin_examples()
    else:
        examples = []
    examples.extend(
        get_examples(num_examples, path=path, max_log_tokens=max_log_tokens)
    )
    if len(diffstrs) == 1:
        return complete(
            context=diffstrs[0],
            prompt=make_prompt(quality=10, hint=hint),
            examples=examples,
            pre_prompt=pre_prompt_commit,
            max_output_tokens=max_output_tokens,
            engine=engine,
            verbose=verbose,
        ).strip()
    else:
        assert len(diffstrs) > 1
        # Process each diff separately and combine the results
        messages = []
        for diffstr in diffstrs:
            message = complete(
                context=diffstr,
                prompt=make_prompt(quality=10, hint=hint),
                examples=examples,
                pre_prompt=pre_prompt_commit,
                max_output_tokens=max_output_tokens,
                engine=engine,
                verbose=verbose,
            ).strip()
            messages.append(message)
        combine_examples = [
            (
                Task(
                    context="Modified func1 to print x + 2.\nAdded a print statement to show the current value of x. This, together with the previous commit, will help with debugging and understanding the function's behavior.",
                    request="Combine the commit messages into a single, concise one-liner.",
                ),
                Result(
                    response="Modified func1 to print x + 2 and added a print statement to show the current value of x. This will help with debugging and understanding the function's behavior."
                ),
            ),
        ]
        combined_message = complete(
            context="\n".join(messages),
            prompt="Combine the commit messages into a single, concise one-liner.",
            examples=combine_examples,
            pre_prompt=default_pre_prompt,
            max_output_tokens=max_output_tokens,
            engine=engine,
            verbose=verbose,
        ).strip()
        return combined_message
