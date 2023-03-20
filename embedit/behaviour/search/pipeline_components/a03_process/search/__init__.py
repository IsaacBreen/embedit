from typing import Literal

import numpy as np

from embedit.behaviour.openai_tools import get_embedding
from embedit.behaviour.openai_tools import get_embeddings
from embedit.structures.embedding import EmbeddedText
from embedit.structures.embedding import EmbeddedTextFileFragment
from embedit.structures.embedding import EmbeddedTextFileFragmentSimilarityResult
from embedit.structures.text_file import TextFileFragment
from embedit.utils.log import logger


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_fragments(fragments: list[TextFileFragment], mode: Literal["openai", "cohere"] = "openai") -> list[EmbeddedTextFileFragment]:
    # Get the embeddings for the fragments
    embeddings = get_embeddings([fragment.contents for fragment in fragments], mode=mode)
    # Return the embedded fragments
    return [EmbeddedTextFileFragment(fragment=fragment, embedding=embedding) for fragment, embedding in
            zip(fragments, embeddings)]


def embed_text(text: str, mode: Literal["openai", "cohere"] = "openai") -> EmbeddedText:
    # Return the embedded text
    return EmbeddedText(text=text, embedding=get_embedding(text, mode=mode))


def get_similarities_for_fragments(
    embedded_text: EmbeddedText,
    embedded_fragments: list[EmbeddedTextFileFragment],
    *,
    threshold: float = 0.0,
) -> list[EmbeddedTextFileFragmentSimilarityResult]:
    logger.info(f"Finding similar fragments from a list of {len(embedded_fragments)} fragments.")
    # Get the embeddings for the fragment
    embedding = embedded_text.embedding
    # Find the most similar fragments
    similarities = [cosine_similarity(embedding, fragment.embedding) for fragment in embedded_fragments]
    logger.info(
        f"Similarity statistics: min={min(similarities):.3f}, max={max(similarities):.3f}, mean={np.mean(similarities):.3f}, median={np.median(similarities):.3f}, std={np.std(similarities):.3f}"
    )
    # Return the most similar fragments
    return [EmbeddedTextFileFragmentSimilarityResult(embedded_fragment=fragment, similarity=similarity) for
            fragment, similarity in
            zip(embedded_fragments, similarities) if similarity >= threshold]
