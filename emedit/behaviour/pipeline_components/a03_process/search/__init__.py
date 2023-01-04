import numpy as np
import openai
from tenacity import wait_random_exponential, stop_after_attempt, retry

from emedit.structures.embedding import EmbeddedTextFileFragment, EmbeddedText, \
    EmbeddedTextFileFragmentSimilarityResult
from emedit.structures.text_file import TextFileFragment


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, engine="text-embedding-ada-002") -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(
    list_of_text: list[str], engine="text-embedding-ada-002"
) -> list[list[float]]:
    assert len(list_of_text) > 0
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.Embedding.create(input=list_of_text, engine=engine).data
    data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
    return [d["embedding"] for d in data]


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
