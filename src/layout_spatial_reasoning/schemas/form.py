"""Dataset item schema."""

from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout


class OrderConstraint(BaseModel):
    """Partial reading-order relation: before must precede after."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before: str = Field(min_length=1)
    after: str = Field(min_length=1)


class OrderConstraintRecord(BaseModel):
    """LLM-extracted reading-order constraints for one form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form_id: str = Field(min_length=1)
    constraints: list[OrderConstraint] = Field(default_factory=list)


class FormSpec(BaseModel):
    """One input form in the Form Layout Benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    form_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    controls: list[Control] = Field(min_length=1)
    order_constraints: list[OrderConstraint] = Field(default_factory=list)
    target_layouts: list[Layout] = Field(default_factory=list)
