from pathlib import Path

from attrs import define


@define(frozen=True)
class TextFile:
    path: str | Path
    contents: str


@define(frozen=True)
class TextFileFragment:
    path: str | Path
    contents: str
    start_line: int

    @property
    def end_line(self) -> int:
        return self.start_line + self.contents.count("\n")
