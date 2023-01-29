import numpy as np

from embedit.behaviour.openai_tools import get_embedding, get_embeddings
from embedit.structures.embedding import EmbeddedTextFileFragment, EmbeddedText, \
    EmbeddedTextFileFragmentSimilarityResult
from embedit.structures.text_file import TextFileFragment


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def embed_fragments(fragments: list[TextFileFragment]) -> list[EmbeddedTextFileFragment]:
    # Get the embeddings for the fragments
    embeddings = get_embeddings([fragment.contents for fragment in fragments])
    # Return the embedded fragments
    return [EmbeddedTextFileFragment(fragment=fragment, embedding=embedding) for fragment, embedding in
        zip(fragments, embeddings)]


def embed_text(text: str) -> EmbeddedText:
    # Return the embedded text
    return EmbeddedText(text=text, embedding=get_embedding(text))


def find_similar_fragments(
    embedded_text: str, embedded_fragments: list[EmbeddedTextFileFragment], *, threshold: float = 0.0, top_n: int = 3,
    verbosity: int = 1
) -> list[EmbeddedTextFileFragmentSimilarityResult]:
    if verbosity > 0:
        print("Finding similar fragments from a list of", len(embedded_fragments), "fragments.")
    # Get the embeddings for the fragment
    embedding = embedded_text.embedding
    # Find the most similar fragments
    similarities = [cosine_similarity(embedding, fragment.embedding) for fragment in embedded_fragments]
    print(
        f"Similarity statistics: min={min(similarities):.3f}, max={max(similarities):.3f}, mean={np.mean(similarities):.3f}, median={np.median(similarities):.3f}, std={np.std(similarities):.3f}"
    )
    # Return the most similar fragments
    return [EmbeddedTextFileFragmentSimilarityResult(embedded_fragment=fragment, similarity=similarity) for
        fragment, similarity in
        zip(embedded_fragments, similarities) if similarity >= threshold][:top_n]
