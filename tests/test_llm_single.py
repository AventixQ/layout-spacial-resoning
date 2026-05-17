from layout_spatial_reasoning.methods.llm_single import (
    build_system_prompt,
    controls_to_json,
    parse_layout_response,
    _response_format,
)
from layout_spatial_reasoning.schemas import Control


def test_parse_layout_response_accepts_json_fence():
    layout = parse_layout_response(
        """```json
        {"sections":[{"section_id":"s001","section_name":"Form","rows":[]}]}
        ```"""
    )

    assert layout.sections[0].section_id == "s001"


def test_controls_to_json_uses_input_schema():
    payload = controls_to_json([Control(id="c01", label="Email", type="text")])

    assert '"id": "c01"' in payload
    assert '"label": "Email"' in payload
    assert '"type": "text"' in payload


def test_build_few_shot_prompt_inserts_examples_placeholder():
    prompt = build_system_prompt("few_shot", examples=[])

    assert "{{EXAMPLES_JSON}}" not in prompt
    assert "[]" in prompt


def test_structured_output_schema_has_required_fields():
    response_format = _response_format("structured_output")
    schema = response_format["json_schema"]["schema"]

    assert schema["required"] == ["sections"]
    section_schema = schema["properties"]["sections"]["items"]
    assert section_schema["required"] == ["section_id", "section_name", "rows"]
