from typing import Literal

from delegatefn import delegate

from embedit.behaviour.search.pipeline_components.a01_gather import gather
from embedit.behaviour.search.pipeline_components.a02_split import split_file
from embedit.behaviour.search.pipeline_components.a03_process.search import embed_fragments
from embedit.behaviour.search.pipeline_components.a03_process.search import embed_text
from embedit.behaviour.search.pipeline_components.a03_process.search import get_similarities_for_fragments
from embedit.structures.embedding import EmbeddedTextFileFragmentSimilarityResult
from embedit.utils.log import logger


@delegate(get_similarities_for_fragments, ignore={"embedded_text", "embedded_fragments"})
def semantic_search(
    query: str,
    *files: str,
    fragment_lines: int = 20,
    min_fragment_lines: int = 0,
    mode: Literal["openai", "cohere"] = "openai",
    **kwargs
) -> list[EmbeddedTextFileFragmentSimilarityResult]:
    assert len(files) > 0, "No files were provided"
    # Gather the files
    files = gather(*files)
    # Split the files
    fragments = [fragment for file in files for fragment in split_file(file, fragment_lines=fragment_lines)]
    fragments = [fragment for fragment in fragments if len(fragment.contents.splitlines()) >= min_fragment_lines]
    # Embed the fragments
    logger.info(f"Embedding {len(fragments)} fragments")
    embedded_fragments = embed_fragments(fragments, mode=mode)
    # Embed the query
    logger.info(f"Embedding the query")
    embedded_query = embed_text(query, mode=mode)
    # Find the most similar fragments
    return get_similarities_for_fragments(embedded_query, embedded_fragments, **kwargs)
