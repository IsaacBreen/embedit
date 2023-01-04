from attrs import define, field


@define(frozen=True)
class TextFile:
    path: str
    contents: str


@define(frozen=True)
class TextFileFragment:
    path: str
    contents: str
    start_line: int

    @property
    def end_line(self) -> int:
        return self.start_line + self.contents.count("\n")