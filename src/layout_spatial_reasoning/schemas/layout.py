"""Output layout schema for a twelve-column grid."""

from pydantic import BaseModel, ConfigDict, Field


class LayoutControl(BaseModel):
    """Position of one input control in the twelve-column grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    colStart: int = Field(ge=1, le=12)
    colSpan: int = Field(ge=1, le=12)


class Row(BaseModel):
    """A row of controls inside one section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    row_id: str = Field(min_length=1)
    controls: list[LayoutControl] = Field(default_factory=list)


class Section(BaseModel):
    """Semantic section of a generated form layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(min_length=1)
    section_name: str = Field(min_length=1)
    rows: list[Row] = Field(default_factory=list)


class Layout(BaseModel):
    """Complete generated layout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: list[Section] = Field(default_factory=list)
