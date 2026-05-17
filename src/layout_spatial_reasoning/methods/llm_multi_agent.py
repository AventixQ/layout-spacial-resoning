"""Multi-agent LLM layout generation."""

import json
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.methods.llm_single import (
    _extract_json_object,
    controls_to_json,
)
from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout
from layout_spatial_reasoning.schemas.validation import validate_layout


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
GROUPING_PROMPT = PROMPT_DIR / "multi_agent_grouping.txt"
NAMING_PROMPT = PROMPT_DIR / "multi_agent_naming.txt"
LAYOUT_PROMPT = PROMPT_DIR / "multi_agent_layout.txt"


class ControlGroup(BaseModel):
    """A semantic group produced by Method 2 Agent 1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str = Field(min_length=1)
    control_ids: list[str] = Field(min_length=1)


class SemanticGrouping(BaseModel):
    """Complete grouping produced by Method 2 Agent 1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: list[ControlGroup] = Field(min_length=1)


class NamedSection(BaseModel):
    """A named semantic section produced by Method 2 Agent 2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section_id: str = Field(min_length=1)
    section_name: str = Field(min_length=1)
    control_ids: list[str] = Field(min_length=1)


class NamedSections(BaseModel):
    """Complete section naming result produced by Method 2 Agent 2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: list[NamedSection] = Field(min_length=1)


def generate_layout(
    controls: list[Control],
    *,
    model: str | None = None,
) -> Layout:
    """Generate a complete form layout through three LLM agents."""
    return generate_layout_openai(controls, model=model)


def generate_layout_openai(
    controls: list[Control],
    *,
    model: str | None = None,
) -> Layout:
    """OpenAI implementation of Method 2."""
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for Method 2.")

    model_name = model or os.environ.get("OPENAI_LAYOUT_MODEL", "gpt-4.1")
    client = OpenAI(api_key=api_key)

    grouping = semantic_grouping_agent(client, controls, model=model_name)
    named_sections = section_naming_agent(
        client,
        controls,
        grouping,
        model=model_name,
    )
    return spatial_layout_agent(
        client,
        controls,
        named_sections,
        model=model_name,
    )


def semantic_grouping_agent(
    client: OpenAI,
    controls: list[Control],
    *,
    model: str,
) -> SemanticGrouping:
    """Agent 1: group controls by semantic relatedness."""
    content = _call_json_model(
        client,
        model=model,
        system_prompt=GROUPING_PROMPT.read_text(encoding="utf-8"),
        payload={"controls": _controls_payload(controls)},
    )
    grouping = SemanticGrouping.model_validate_json(content)
    validate_semantic_grouping(controls, grouping)
    return grouping


def section_naming_agent(
    client: OpenAI,
    controls: list[Control],
    grouping: SemanticGrouping,
    *,
    model: str,
) -> NamedSections:
    """Agent 2: assign human-readable section names to fixed groups."""
    content = _call_json_model(
        client,
        model=model,
        system_prompt=NAMING_PROMPT.read_text(encoding="utf-8"),
        payload={
            "controls": _controls_payload(controls),
            "groups": grouping.model_dump(mode="json")["groups"],
        },
    )
    named_sections = NamedSections.model_validate_json(content)
    validate_named_sections(grouping, named_sections)
    return named_sections


def spatial_layout_agent(
    client: OpenAI,
    controls: list[Control],
    named_sections: NamedSections,
    *,
    model: str,
) -> Layout:
    """Agent 3: arrange named sections into the final grid layout."""
    content = _call_json_model(
        client,
        model=model,
        system_prompt=LAYOUT_PROMPT.read_text(encoding="utf-8"),
        payload={
            "controls": _controls_payload(controls),
            "sections": named_sections.model_dump(mode="json")["sections"],
        },
    )
    layout = Layout.model_validate_json(content)
    errors = validate_layout(controls, layout)
    if errors:
        raise ValueError(f"Invalid Method 2 layout: {'; '.join(errors)}")
    return layout


def validate_semantic_grouping(
    controls: list[Control],
    grouping: SemanticGrouping,
) -> None:
    """Raise when Agent 1 output does not partition input controls exactly once."""
    expected = {control.id for control in controls}
    actual = [
        control_id
        for group in grouping.groups
        for control_id in group.control_ids
    ]
    _raise_partition_errors(expected, actual, "semantic grouping")

    duplicated_group_ids = _duplicates(group.group_id for group in grouping.groups)
    if duplicated_group_ids:
        raise ValueError(f"Duplicated group ids: {duplicated_group_ids}")


def validate_named_sections(
    grouping: SemanticGrouping,
    named_sections: NamedSections,
) -> None:
    """Raise when Agent 2 output changes the groups instead of only naming them."""
    expected_groups = {
        group.group_id: tuple(group.control_ids)
        for group in grouping.groups
    }
    actual_groups = {
        section.section_id: tuple(section.control_ids)
        for section in named_sections.sections
    }

    if expected_groups != actual_groups:
        raise ValueError("Named sections must preserve Agent 1 group ids and controls.")

    duplicated_section_ids = _duplicates(
        section.section_id for section in named_sections.sections
    )
    if duplicated_section_ids:
        raise ValueError(f"Duplicated section ids: {duplicated_section_ids}")


def parse_semantic_grouping_response(content: str) -> SemanticGrouping:
    """Parse an Agent 1 response."""
    return SemanticGrouping.model_validate_json(_extract_json_object(content))


def parse_named_sections_response(content: str) -> NamedSections:
    """Parse an Agent 2 response."""
    return NamedSections.model_validate_json(_extract_json_object(content))


def controls_and_groups_to_json(
    controls: list[Control],
    grouping: SemanticGrouping,
) -> str:
    """Serialize Agent 2 payload using thesis input controls and Agent 1 groups."""
    return json.dumps(
        {
            "controls": _controls_payload(controls),
            "groups": grouping.model_dump(mode="json")["groups"],
        },
        ensure_ascii=False,
    )


def controls_and_sections_to_json(
    controls: list[Control],
    named_sections: NamedSections,
) -> str:
    """Serialize Agent 3 payload using input controls and named sections."""
    return json.dumps(
        {
            "controls": _controls_payload(controls),
            "sections": named_sections.model_dump(mode="json")["sections"],
        },
        ensure_ascii=False,
    )


def _call_json_model(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    payload: dict[str, object],
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Method 2 agent returned an empty response.")
    return _extract_json_object(content)


def _controls_payload(controls: list[Control]) -> list[dict[str, str]]:
    return json.loads(controls_to_json(controls))["controls"]


def _raise_partition_errors(
    expected: set[str],
    actual: list[str],
    context: str,
) -> None:
    missing = sorted(expected.difference(actual))
    unknown = sorted(set(actual).difference(expected))
    duplicated = _duplicates(actual)

    errors = []
    if missing:
        errors.append(f"missing controls: {missing}")
    if unknown:
        errors.append(f"unknown controls: {unknown}")
    if duplicated:
        errors.append(f"duplicated controls: {duplicated}")

    if errors:
        raise ValueError(f"Invalid {context}: {'; '.join(errors)}")


def _duplicates(values) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)
