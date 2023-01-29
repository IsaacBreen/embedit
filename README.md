# embedit

`embedit` is a command line tool for performing bulk operations on text files (e.g. a package) with OpenAI's API. It currently provides two commands: `search`, which performs semantic search on text files using embeddings, and `transform` which performs arbitrary transformations using a custom prompt.

*Can't I just feed my files to the API directly?*

You could. But transforming each file independently could lead to inconsistent behaviour. `embedit transform` combines your files into a single prompt so that they can be transformed in a coherent way and then splits the result back into individual files.

## Installation

Install `embedit` using `pip`:

```bash
pip install embedit
```

This will install `embedit` and its dependencies, including `openai`. You will also need to set the `OPENAI_API_KEY` environment variable to your OpenAI API key if you haven't already done so.

## Usage

`embedit` provides two commands: `search` and `transform`.

### Search

`embedit search` performs semantic searches on text files. You specify a search query and a list of text files to search, `embedit` will fetch text from the files, split them into segments, embed them using OpenAI's API, and print them out in order of cosine distance to the query.

```bash
embedit search "search query" file1.txt file2.txt ...
```

#### Options

- `--order`: the order in which the results should be displayed (ascending or descending by similarity score). Default: `ascending`.

- `--top-n`: the number of top results to display. Default: `3`.

- `--threshold`: a similarity threshold below which results should be filtered out. Default: `0.0`.

- `--frament_lines`: the target fragment length in number of lines. Default: `10`.

- `--min_fragment_lines`: the minimum fragment length in number of lines. Default: `0`.

### Transform

The `transform` command allows you to transform one or more text files by passing their markdown representation with a given prompt to the OpenAI API.

```bash
embedit transform **/*.py --prompt "Add a docstring at the top of each file" --output-dir out
```

#### Options

- `--files`: One or more text files to transform.
- `--transformation_fn`: The function to apply on the files.
- `--output_dir` : The directory to save the transformed files.
- `--yes`: Don't prompt before creating or overwriting files.
- `--engine`: The OpenAI API engine to use.
  - Defaults to 'text-davinci-003'; however, if you have access to "code-davinci-002", I recommend using that instead.
- `--verbose`: Whether to print verbose output.
- `--max_chunk_len`: The maximum length (in characters) of chunks to pass to the OpenAI API.

### Generate commit message

The `commit-msg` command will generate a commit message based on the diff of the staged files and the commit history. 

To use it, you can run it directly:

```bash
embedit commit-msg
```

To generate and commit the changes in one step, you can use the `autocommit` command:

```bash
embedit autocommit
```

I haven't tried to add `commit-msg` as a git hook, but I imagine it would work.

#### Options

- `--path`: The path to diff against.
- `--max-log-tokens`: The maximum number of tokens to include in the commit message.
- `--max-diff-tokens`: The maximum number of tokens to include in the diff.
- `--max-output-tokens`: The maximum number of tokens to include in the OpenAI API output.
- `--engine`: The OpenAI API engine to use.
- `--num-examples`: The number of examples to use.
- `--use-builtin-examples`: Whether to use the built-in examples.
- `--hint`: A hint to pass to the OpenAI API.
- `--verbose`: Print verbose output.
- `--git-params`: Keyword arguments to pass to the git commit command.


For example, the below command will generate a commit message using `code-davinci-002`, passing a hint that the document parameters have been updated, and will use not any of your previous commits as examples. The latter option is useful if your past commit messages have suffered *neglect*.

```bash
embedit autocommit --engine "code-davinci-002" --hint "doc params" --num-examples 0
```
## Tips

### Wildcards
You can also use wildcards to specify a pattern of files to search in. Here's an example of how you can use the `**` wildcard to search for Python files in all directories in the current directory and its subdirectories:

```bash
embedit search "query" **/*.py
```

Bear in mind that the behavior of the `*` and `**` wildcards may vary depending on your operating system and the terminal shell you're using.

### Autocommit workflow

I like to use the following alias, `qc` (quick commit) to automatically generate and commit changes:

```bash
alias qc="embedit autocommit --num-examples 0 --engine \"code-davinci-002\""
```

Example usage:

```bash
qc -h "hint hint"
```

Then, when it gets something wrong, I edit the commit message and run `git commit --amend` to fix it.

## Contributing

If you find a bug or want to contribute to the development of `embedit`, you can create a new issue or submit a pull request.

## License

`embedit` is released under the MIT license. Do whatever you want with it.