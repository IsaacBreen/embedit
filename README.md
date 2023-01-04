# emtool

`emtool` is a command-line tool that uses OpenAI's embedding API to 

## Installation

`emtool` requires Python 3.11 or higher.

Install the package and its dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

## Usage

To search for semantic similarity between a query and a set of text files, run the `emtool` command followed by the query and the paths to the text files.

```bash
emtool "search query" file1.txt file2.txt /path/to/dir
```

Directories can also be provided as arguments, but they will be ignored.

### Options

- `--order`: The order in which to sort the results. Can be either "ascending" or "descending" (default is "ascending").
- `--top_n`: The number of top results to display (default is 3).
- `--threshold`: The minimum similarity score for a result to be displayed (default is 0.0).

## Example

```bash
emtool "search query" file1.txt file2.txt --order=descending --top_n=5 --threshold=0.5
```

This will search for the query "search query" in `file1.txt` and `file2.txt`, and display the top 5 results with a similarity score greater than or equal to 0.5, sorted in descending order by similarity score.

To use the 'emtool' script, you will need to provide the query and at least one text file to search in. The script will then use the 'semantic_search' function to find the most semantically similar fragments within the provided text files. The results will be sorted in either ascending or descending order (based on the 'order' argument, which has a default value of '"ascending"'), and the top `n` results will be displayed (based on the `top_n` argument, which has a default value of `3`). A similarity threshold can also be set with the `threshold` argument (default is `0.0`), which filters out results with a lower similarity score.

The results will be printed to the console, along with the similarity score and file path for each result. The contents of the result will also be highlighted with appropriate syntax highlighting.

For example, running the following command:

```bash
emtool "search query" file1.txt file2.txt --order=descending --top_n=5 --threshold=0.5
```

Will search for the query "search query" in file1.txt and file2.txt, and display the top 5 results with a similarity score greater than or equal to 0.5, sorted in descending order by similarity score.

Dependencies

`emtool` is built on top of the following libraries:

- [openai](https://github.com/openai/openai)
- [delegatefn](https://github.com/isaacbreen/delegatefn
- [rich](https://github.com/willmcgugan/rich)
