"""Input form control schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ControlType = Literal[
    "text",
    "long_text",
    "choice",
    "multichoice",
    "boolean",
    "date",
    "number",
    "file",
]


class Control(BaseModel):
    """Single input form control from the thesis input schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    type: ControlType
    help_text: str = ""
