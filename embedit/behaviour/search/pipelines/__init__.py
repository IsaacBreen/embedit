from delegatefn import delegate

from embedit.behaviour.search.pipeline_components.a01_gather import gather
from embedit.behaviour.search.pipeline_components.a02_split import split_file
from embedit.behaviour.search.pipeline_components.a03_process.search import embed_text, find_similar_fragments, embed_fragments
from embedit.structures.embedding import EmbeddedTextFileFragmentSimilarityResult


@delegate(find_similar_fragments, ignore=["embedded_text", "embedded_fragments", "verbosity"])
def semantic_search(
    query: str, *files: str, verbosity: int = 1, fragment_lines: int = 10, min_fragment_lines: int = 0, **kwargs
) -> list[EmbeddedTextFileFragmentSimilarityResult]:
    assert len(files) > 0, "No files were provided"
    # Gather the files
    files = gather(*files)
    # Split the files
    fragments = [fragment for file in files for fragment in split_file(file, fragment_lines=fragment_lines)]
    fragments = [fragment for fragment in fragments if len(fragment.contents.splitlines()) >= min_fragment_lines]
    if verbosity > 0:
        print(f"Embedding {len(fragments)} fragments")
    # Embed the fragments
    embedded_fragments = embed_fragments(fragments)
    if verbosity > 0:
        print(f"Embedding the query")
    # Embed the query
    embedded_query = embed_text(query)
    # Find the most similar fragments
    return find_similar_fragments(embedded_query, embedded_fragments, verbosity=verbosity, **kwargs)
