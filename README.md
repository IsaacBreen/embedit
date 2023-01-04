# emedit

`emedit` is a command line tool for performing semantic searches on a set of text files. It allows you to specify a search query and a list of text files to search, and returns a list of results ranked by their similarity to the query.

## Installation

Install `emedit` using `pip`:

```bash
pip install emedit
```

## Usage

To use `emedit`, run the following command and specify your search query and the list of text files you want to search:

```bash
emedit search "search query" file1.txt file2.txt ...
```
You can also specify the following optional arguments:


`--order`: the order in which the results should be displayed (ascending or descending by similarity score). Default: `ascending`.
`--top-n`: the number of top results to display. Default: `3`.
`--threshold`: a similarity threshold below which results should be filtered out. Default: `0.0`.
`--frament_lines`: the target fragment length in number of lines. Default: `10`.
`--min_fragment_lines`: the minimum fragment length in number of lines. Default: `0`.

## Examples

Here is an example of a search for the query "machine learning" in a set of text files:

```bash
emedit search "machine learning" file1.txt file2.txt file3.txt
This will display the top 3 results for the search, ranked in descending order by similarity score.
```

You can also specify a different number of top results to display using the `--top-n` argument:

```bash
emedit search "machine learning" file1.txt file2.txt file3.txt --top-n 5
```

This will display the top 5 results for the search.

You can also specify a different order for the results using the `--order` argument:

```bash
emedit search "machine learning" file1.txt file2.txt file3.txt --order ascending
```

This will display the results in ascending order by similarity score.

Finally, you can specify a similarity threshold to filter out results below a certain similarity score using the `--threshold` argument:

```bash
emedit search "machine learning" file1.txt file2.txt file3.txt --threshold 0.5
```

This will only display results with a similarity score greater than or equal to 0.5.

### Wildcards
 

You can lso use wildcards to specify a pattern of files to search in. Here's an example of how you can use the `**` wildcard to search for Python files in all directories in the current directory and its subdirectories:

```bash
emedit search "query" **/*.py
```

Bear in mind that the behavior of the `*` and `**` wildcards may vary depending on your operating system and the terminal shell you're using.

## Contributing

If you find a bug or want to contribute to the development of `emedit`, you can create a new issue or submit a pull request.

## License

`emedit` is released under the MIT license. Do whatever you want with it.