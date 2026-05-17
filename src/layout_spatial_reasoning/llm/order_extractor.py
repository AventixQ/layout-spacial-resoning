"""LLM OrderExtractor for reading-order constraints."""

import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.form import OrderConstraint


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "order_extractor.txt"


class _RawOrderConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    before: str = Field(min_length=1)
    after: str = Field(min_length=1)


class _RawOrderConstraintResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: list[_RawOrderConstraint] = Field(default_factory=list)


def extract_order_constraints_openai(
    controls: list[Control],
    *,
    model: str | None = None,
) -> list[OrderConstraint]:
    """Extract reading-order constraints with an OpenAI chat model."""
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OrderExtractor.")

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    model_name = model or os.environ.get("OPENAI_ORDER_MODEL", "gpt-4.1-mini")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": _controls_payload(controls)},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OrderExtractor returned an empty response.")

    raw_response = _RawOrderConstraintResponse.model_validate_json(content)
    raw_constraints = [
        OrderConstraint(before=item.before, after=item.after)
        for item in raw_response.constraints
    ]
    return validate_order_constraints(raw_constraints, controls)


def validate_order_constraints(
    constraints: list[OrderConstraint],
    controls: list[Control],
) -> list[OrderConstraint]:
    """Remove invalid, duplicated, reflexive, and conflicting constraints."""
    control_ids = {control.id for control in controls}
    validated: list[OrderConstraint] = []
    seen: set[tuple[str, str]] = set()

    for constraint in constraints:
        pair = (constraint.before, constraint.after)
        reverse_pair = (constraint.after, constraint.before)
        if constraint.before not in control_ids or constraint.after not in control_ids:
            continue
        if constraint.before == constraint.after:
            continue
        if pair in seen or reverse_pair in seen:
            continue
        seen.add(pair)
        validated.append(constraint)

    return validated


def _controls_payload(controls: list[Control]) -> str:
    return json.dumps(
        {
            "controls": [
                {
                    "id": control.id,
                    "label": control.label,
                    "type": control.type,
                    "help_text": control.help_text,
                }
                for control in controls
            ]
        },
        ensure_ascii=False,
    )
