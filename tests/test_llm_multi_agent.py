import pytest

from layout_spatial_reasoning.methods.llm_multi_agent import (
    ControlGroup,
    NamedSection,
    NamedSections,
    SemanticGrouping,
    controls_and_groups_to_json,
    controls_and_sections_to_json,
    parse_named_sections_response,
    parse_semantic_grouping_response,
    validate_named_sections,
    validate_semantic_grouping,
)
from layout_spatial_reasoning.schemas import Control


def test_parse_semantic_grouping_response_accepts_json_fence():
    grouping = parse_semantic_grouping_response(
        """```json
        {"groups":[{"group_id":"g001","control_ids":["c01","c02"]}]}
        ```"""
    )

    assert grouping.groups[0].group_id == "g001"
    assert grouping.groups[0].control_ids == ["c01", "c02"]


def test_parse_named_sections_response_accepts_json_fence():
    sections = parse_named_sections_response(
        """```json
        {"sections":[{"section_id":"g001","section_name":"Contact","control_ids":["c01"]}]}
        ```"""
    )

    assert sections.sections[0].section_name == "Contact"


def test_validate_semantic_grouping_requires_exact_control_partition():
    controls = [
        Control(id="c01", label="Email", type="text"),
        Control(id="c02", label="Phone", type="text"),
    ]
    grouping = SemanticGrouping(
        groups=[
            ControlGroup(group_id="g001", control_ids=["c01", "c01", "c99"]),
        ]
    )

    with pytest.raises(ValueError, match="missing controls"):
        validate_semantic_grouping(controls, grouping)


def test_validate_named_sections_preserves_agent_one_groups():
    grouping = SemanticGrouping(
        groups=[
            ControlGroup(group_id="g001", control_ids=["c01", "c02"]),
        ]
    )
    named_sections = NamedSections(
        sections=[
            NamedSection(
                section_id="g001",
                section_name="Contact",
                control_ids=["c02", "c01"],
            )
        ]
    )

    with pytest.raises(ValueError, match="preserve Agent 1"):
        validate_named_sections(grouping, named_sections)


def test_agent_payload_serializers_use_expected_shapes():
    controls = [Control(id="c01", label="Email", type="text")]
    grouping = SemanticGrouping(
        groups=[ControlGroup(group_id="g001", control_ids=["c01"])]
    )
    named_sections = NamedSections(
        sections=[
            NamedSection(
                section_id="g001",
                section_name="Contact",
                control_ids=["c01"],
            )
        ]
    )

    grouping_payload = controls_and_groups_to_json(controls, grouping)
    layout_payload = controls_and_sections_to_json(controls, named_sections)

    assert '"controls"' in grouping_payload
    assert '"groups"' in grouping_payload
    assert '"sections"' in layout_payload
    assert '"section_name": "Contact"' in layout_payload
