from pathlib import Path

from layout_spatial_reasoning.dataset import (
    load_forms_jsonl,
    load_generated_layouts_jsonl,
    write_layouts_jsonl,
)
from layout_spatial_reasoning.methods.sequential_baseline import generate_layout


def test_load_sample_forms():
    forms = load_forms_jsonl("data/processed/sample_forms.jsonl")

    assert len(forms) == 3
    assert forms[0].form_id == "contact_basic"


def test_write_and_load_generated_layouts(tmp_path: Path):
    forms = load_forms_jsonl("data/processed/sample_forms.jsonl")
    output_path = tmp_path / "layouts.jsonl"
    layout = generate_layout(forms[0].controls)

    write_layouts_jsonl([(forms[0].form_id, "sequential", layout)], output_path)
    records = load_generated_layouts_jsonl(output_path)

    assert len(records) == 1
    assert records[0].method == "sequential"
