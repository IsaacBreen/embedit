# emedit

`emedit` is a command line tool for performing semantic searches on text files. You specify a search query and a list of text files to search, `emedit` will fetch text from the files, split them into segments, embed them using OpenAI's API, and print them out in order of cosine distance to the query.

## Installation

Install `emedit` using `pip`:

```bash
pip install emedit
```

This will install `emedit` and its dependencies, including `openai`. You will also need to set the `OPENAI_API_KEY` environment variable to your OpenAI API key if you haven't already done so.

## Usage

To use `emedit`, run the following command:

```bash
emedit search "search query" file1.txt file2.txt ...
```
You can also specify the following optional arguments:


- `--order`: the order in which the results should be displayed (ascending or descending by similarity score). Default: `ascending`.

- `--top-n`: the number of top results to display. Default: `3`.

- `--threshold`: a similarity threshold below which results should be filtered out. Default: `0.0`.

- `--frament_lines`: the target fragment length in number of lines. Default: `10`.

- `--min_fragment_lines`: the minimum fragment length in number of lines. Default: `0`.

You can also use wildcards to specify a pattern of files to search in. Here's an example of how you can use the `**` wildcard to search for Python files in all directories in the current directory and its subdirectories:

```bash
emedit search "query" **/*.py
```

Bear in mind that the behavior of the `*` and `**` wildcards may vary depending on your operating system and the terminal shell you're using.

## Contributing

If you find a bug or want to contribute to the development of `emedit`, you can create a new issue or submit a pull request.

## License

`emedit` is released under the MIT license. Do whatever you want with it.