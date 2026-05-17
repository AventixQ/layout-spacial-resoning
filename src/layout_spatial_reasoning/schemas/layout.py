"""Output layout schema for a twelve-column grid."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LayoutControl:
    id: str
    colStart: int
    colSpan: int


@dataclass(frozen=True)
class Row:
    row_id: str
    controls: list[LayoutControl] = field(default_factory=list)


@dataclass(frozen=True)
class Section:
    section_id: str
    section_name: str
    rows: list[Row] = field(default_factory=list)


@dataclass(frozen=True)
class Layout:
    sections: list[Section] = field(default_factory=list)
