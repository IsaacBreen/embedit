from attrs import define

from .text_file import TextFileFragment


@define(frozen=True)
class EmbeddedTextFileFragment:
    fragment: TextFileFragment
    embedding: list[float]


@define(frozen=True)
class EmbeddedText:
    text: str
    embedding: list[float]


@define(frozen=True)
class EmbeddedTextFileFragmentSimilarityResult:
    embedded_fragment: EmbeddedTextFileFragment
    similarity: float
