from delegatefn import delegate

from emtool.behaviour.pipeline_components.a01_gather import gather
from emtool.behaviour.pipeline_components.a02_split import split_file
from emtool.behaviour.pipeline_components.a03_process.search import embed_text, find_similar_fragments, embed_fragments
from emtool.behaviour.pipeline_components.a04_combine import combine_fragments
from emtool.structures.embedding import EmbeddedTextFileFragment, EmbeddedTextFileFragmentSimilarityResult


@delegate(find_similar_fragments, ignore=["embedded_text", "embedded_fragments", "verbosity"])
def semantic_search(query: str, *files: str, verbosity: int = 1, **kwargs) -> list[EmbeddedTextFileFragmentSimilarityResult]:
    assert len(files) > 0, "No files were provided"
    # Gather the files
    files = gather(*files)
    # Split the files
    fragments = [fragment for file in files for fragment in split_file(file)]
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