from layout_spatial_reasoning.llm.providers import _openai_extra_body_for_response_format
from layout_spatial_reasoning.methods.llm_single import _layout_json_schema


def test_openai_response_format_can_be_mapped_to_vllm_guided_json():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "form_layout",
            "strict": True,
            "schema": _layout_json_schema(),
        },
    }

    extra_body = _openai_extra_body_for_response_format(
        response_format,
        "vllm_guided_json",
    )

    assert extra_body == {"guided_json": response_format["json_schema"]["schema"]}


def test_openai_response_format_can_be_mapped_to_vllm_structured_outputs():
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "form_layout",
            "strict": True,
            "schema": _layout_json_schema(),
        },
    }

    extra_body = _openai_extra_body_for_response_format(
        response_format,
        "vllm_structured_outputs",
    )

    assert extra_body == {
        "structured_outputs": {"json": response_format["json_schema"]["schema"]}
    }
