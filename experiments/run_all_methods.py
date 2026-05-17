"""Run all layout generation methods on a selected split."""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset import load_forms_jsonl, write_layouts_jsonl
from layout_spatial_reasoning.methods.fine_tuned_model import (
    generate_layout as fine_tuned_layout,
)
from layout_spatial_reasoning.methods.graph_community import (
    generate_layout_from_env as graph_layout,
)
from layout_spatial_reasoning.methods.hybrid import generate_layout_from_env as hybrid_layout
from layout_spatial_reasoning.methods.llm_multi_agent import (
    generate_layout as llm_multi_agent_layout,
)
from layout_spatial_reasoning.methods.llm_single import generate_layout as llm_single_layout
from layout_spatial_reasoning.methods.random_baseline import generate_layout as random_layout
from layout_spatial_reasoning.methods.sequential_baseline import (
    generate_layout as sequential_layout,
)


METHODS = {
    "sequential": sequential_layout,
    "random": random_layout,
    "graph_community": graph_layout,
}


def main() -> None:
    input_path = Path("data/processed/sample_forms.jsonl")
    output_path = Path("outputs/generated_layouts/sample_baselines.jsonl")

    forms = load_forms_jsonl(input_path)
    records = []
    for form_index, form in enumerate(forms, start=1):
        for method_name, method in METHODS.items():
            if method_name == "random":
                layout = method(form.controls, seed=form_index)
            else:
                layout = method(form.controls)
            records.append((form.form_id, method_name, layout))

        if os.environ.get("INCLUDE_LLM_SINGLE", "false").lower() == "true":
            for variant in ["zero_shot", "few_shot", "cot", "structured_output"]:
                layout = llm_single_layout(form.controls, variant=variant)
                records.append((form.form_id, f"llm_single_{variant}", layout))

        if os.environ.get("INCLUDE_LLM_MULTI_AGENT", "false").lower() == "true":
            layout = llm_multi_agent_layout(form.controls)
            records.append((form.form_id, "llm_multi_agent", layout))

        if os.environ.get("INCLUDE_FINE_TUNED_MODEL", "false").lower() == "true":
            layout = fine_tuned_layout(form.controls)
            records.append((form.form_id, "fine_tuned_model", layout))

        if os.environ.get("INCLUDE_HYBRID", "false").lower() == "true":
            layout = hybrid_layout(form.controls)
            records.append((form.form_id, "hybrid", layout))

    write_layouts_jsonl(records, output_path)
    print(f"Wrote {len(records)} generated layouts to {output_path}.")


if __name__ == "__main__":
    main()
