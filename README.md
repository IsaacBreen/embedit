# emtool

`emtool` is a tool for searching for semantically similar text in a set of files. It uses OpenAI's GPT-3 model to understand the meaning of the search query and the contents of the files, and returns a list of results sorted by similarity.

## Installation

Install `emtool` using Poetry:

```
$ poetry install
```

## Usage

To use `emtool`, simply call the `emtool` script with a search query and a list of files or directories to search in:

```
$ emtool "search query" file1.txt file2.txt directory/
```

By default, results are sorted in descending order by similarity. To sort in ascending order, use the `--order ascending` flag:

```
$ emtool "search query" file1.txt file2.txt --order ascending
```

Additional options can be passed to the underlying semantic search pipeline by using keyword arguments. For example, to increase the number of tokens to compare in each file fragment, use the `--fragment-length` option:

```
$ emtool "search query" file1.txt file2.txt --fragment-length 20
```

For a full list of options, run `emtool --help`.
