"""Input form control schema."""

from dataclasses import dataclass
from typing import Literal

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


@dataclass(frozen=True)
class Control:
    id: str
    label: str
    type: ControlType
    help_text: str = ""
