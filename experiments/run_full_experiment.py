"""Run the complete thesis experiment pipeline."""

import csv
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.dataset import (
    load_forms_jsonl,
    load_order_constraints_jsonl,
    write_layouts_jsonl,
)
from layout_spatial_reasoning.embeddings.provider import embedding_function_from_env
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout
from layout_spatial_reasoning.evaluation.statistical_tests import run_statistical_tests
from layout_spatial_reasoning.llm.providers import (
    PROVIDER_ENV_KEYS,
    model_from_env,
    normalize_provider,
)
from layout_spatial_reasoning.methods.fine_tuned_model import (
    generate_layout as fine_tuned_layout,
)
from layout_spatial_reasoning.methods.graph_community import (
    generate_layout_from_env as graph_layout,
)
from layout_spatial_reasoning.methods.hybrid import generate_layout as hybrid_layout
from layout_spatial_reasoning.methods.llm_multi_agent import (
    generate_layout as llm_multi_agent_layout,
)
from layout_spatial_reasoning.methods.llm_single import (
    PromptVariant,
    generate_layout as llm_single_layout,
)
from layout_spatial_reasoning.methods.random_baseline import generate_layout as random_layout
from layout_spatial_reasoning.methods.sequential_baseline import (
    generate_layout as sequential_layout,
)


SINGLE_VARIANTS: list[PromptVariant] = [
    "zero_shot",
    "few_shot",
    "cot",
    "structured_output",
]


def main() -> None:
    load_env()
    run_id = os.environ.get("EXPERIMENT_RUN_ID", "latest")
    output_root = Path(os.environ.get("EXPERIMENT_OUTPUT_ROOT", "outputs/experiments"))
    run_dir = output_root / run_id
    generated_dir = run_dir / "generated_layouts"
    errors_dir = run_dir / "errors"
    metrics_dir = run_dir / "metrics"
    stats_dir = run_dir / "statistical_tests"

    forms_path = Path(os.environ.get("FORMS_PATH", "data/processed/sample_forms.jsonl"))
    order_constraints_path = Path(
        os.environ.get("ORDER_CONSTRAINTS_PATH", "data/processed/order_constraints.jsonl")
    )
    forms = load_forms_jsonl(forms_path)
    order_constraints = (
        load_order_constraints_jsonl(order_constraints_path)
        if order_constraints_path.exists()
        else {}
    )

    layout_records = []
    errors = []
    latency_rows = []
    plans = build_experiment_plan()
    for plan in plans:
        records, plan_errors, plan_latency = run_generation_plan(forms, plan)
        layout_records.extend(records)
        errors.extend(plan_errors)
        latency_rows.extend(plan_latency)
        write_layouts_jsonl(
            records,
            generated_dir / f"{plan['method']}.jsonl",
        )
        write_jsonl(plan_errors, errors_dir / f"{plan['method']}_errors.jsonl")

    write_layouts_jsonl(layout_records, generated_dir / "all_generated_layouts.jsonl")
    write_jsonl(errors, errors_dir / "all_errors.jsonl")
    write_jsonl(latency_rows, metrics_dir / "latency.jsonl")

    metrics = evaluate_records(forms, order_constraints, layout_records)
    metrics_path = metrics_dir / "metrics.csv"
    write_metrics_csv(metrics, metrics_path)

    summary = summarize_metrics(metrics)
    summary.to_csv(metrics_dir / "method_summary.csv", index=False)
    (metrics_dir / "method_summary.md").write_text(
        to_markdown_table(summary),
        encoding="utf-8",
    )

    stats = run_statistical_tests(pd.DataFrame([record.model_dump() for record in metrics]))
    (stats_dir / "statistical_tests.json").parent.mkdir(parents=True, exist_ok=True)
    (stats_dir / "statistical_tests.json").write_text(
        json.dumps(stats, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "forms_path": str(forms_path),
        "plan_count": len(plans),
        "layout_count": len(layout_records),
        "error_count": len(errors),
        "output_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote full experiment outputs to {run_dir}.")


def build_experiment_plan() -> list[dict[str, str]]:
    """Build the full provider/method execution plan from environment config."""
    plans: list[dict[str, str]] = [
        {"method": "sequential", "provider": "local", "model": ""},
        {"method": "random", "provider": "local", "model": ""},
        {"method": "graph_community", "provider": "local", "model": ""},
    ]

    providers = [
        normalize_provider(provider)
        for provider in os.environ.get(
            "EXPERIMENT_LLM_PROVIDERS",
            "openai,gemini,claude",
        ).split(",")
        if provider.strip()
    ]
    for provider in providers:
        model = model_from_env(provider)
        for variant in SINGLE_VARIANTS:
            plans.append(
                {
                    "method": f"{provider}_{slug(model)}_llm_single_{variant}",
                    "provider": provider,
                    "model": model,
                    "kind": "llm_single",
                    "variant": variant,
                }
            )
        plans.append(
            {
                "method": f"{provider}_{slug(model)}_llm_multi_agent",
                "provider": provider,
                "model": model,
                "kind": "llm_multi_agent",
            }
        )
        plans.append(
            {
                "method": f"{provider}_{slug(model)}_hybrid",
                "provider": provider,
                "model": model,
                "kind": "hybrid",
            }
        )

    if os.environ.get("INCLUDE_FINE_TUNED_MODEL", "false").lower() == "true":
        model = os.environ.get("FINE_TUNED_LAYOUT_MODEL", "")
        plans.append(
            {
                "method": f"openai_{slug(model)}_fine_tuned_model",
                "provider": "openai",
                "model": model,
                "kind": "fine_tuned_openai",
            }
        )

    if os.environ.get("INCLUDE_LOCAL_LORA", "false").lower() == "true":
        model = os.environ.get("LORA_OUTPUT_DIR", "outputs/lora/method3")
        plans.append(
            {
                "method": f"local_{slug(model)}_fine_tuned_model",
                "provider": "local",
                "model": model,
                "kind": "fine_tuned_local",
            }
        )

    return plans


def run_generation_plan(forms, plan: dict[str, str]):
    """Run one method/provider plan over all forms and collect errors."""
    records = []
    errors = []
    latency_rows = []
    if plan["provider"] in PROVIDER_ENV_KEYS and not os.environ.get(
        PROVIDER_ENV_KEYS[plan["provider"]]
    ):
        error = {
            "method": plan["method"],
            "provider": plan["provider"],
            "model": plan["model"],
            "error_type": "MissingApiKey",
            "error": f"{PROVIDER_ENV_KEYS[plan['provider']]} is not set.",
        }
        return records, [error], latency_rows

    for form_index, form in enumerate(forms, start=1):
        started_at = perf_counter()
        try:
            layout = generate_for_plan(form.controls, form_index, plan)
        except Exception as error:  # noqa: BLE001 - experiment runner records failures.
            errors.append(
                {
                    "form_id": form.form_id,
                    "method": plan["method"],
                    "provider": plan["provider"],
                    "model": plan["model"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        latency_rows.append(
            {
                "form_id": form.form_id,
                "method": plan["method"],
                "latency_seconds": perf_counter() - started_at,
            }
        )
        records.append((form.form_id, plan["method"], layout))
    return records, errors, latency_rows


def generate_for_plan(controls, form_index: int, plan: dict[str, str]):
    method = plan["method"]
    if method == "sequential":
        return sequential_layout(controls)
    if method == "random":
        return random_layout(controls, seed=form_index)
    if method == "graph_community":
        return graph_layout(controls)

    kind = plan["kind"]
    if kind == "llm_single":
        return llm_single_layout(
            controls,
            variant=plan["variant"],
            provider=plan["provider"],
            model=plan["model"],
        )
    if kind == "llm_multi_agent":
        return llm_multi_agent_layout(
            controls,
            provider=plan["provider"],
            model=plan["model"],
        )
    if kind == "hybrid":
        return hybrid_layout(
            controls,
            provider=plan["provider"],
            model=plan["model"],
        )
    if kind == "fine_tuned_openai":
        return fine_tuned_layout(controls, provider="openai", model=plan["model"])
    if kind == "fine_tuned_local":
        return fine_tuned_layout(controls, provider="local", model=plan["model"])
    raise ValueError(f"Unsupported experiment plan: {plan}")


def evaluate_records(forms, order_constraints, layout_records):
    form_by_id = {form.form_id: form for form in forms}
    embedding_function = embedding_function_from_env()
    metrics = []
    for form_id, method, layout in layout_records:
        form = form_by_id[form_id]
        if form_id in order_constraints:
            form = form.model_copy(update={"order_constraints": order_constraints[form_id]})
        generated = type("Generated", (), {"form_id": form_id, "method": method, "layout": layout})
        metrics.append(
            evaluate_generated_layout(
                form,
                generated,
                embedding_function=embedding_function,
            )
        )
    return metrics


def summarize_metrics(metrics) -> pd.DataFrame:
    frame = pd.DataFrame([record.model_dump() for record in metrics])
    if frame.empty:
        return pd.DataFrame()
    return (
        frame.groupby("method", as_index=False)
        .agg(
            form_count=("form_id", "count"),
            mean_grid_utilization=("grid_utilization", "mean"),
            mean_semantic_coherence_row=("semantic_coherence_row", "mean"),
            mean_semantic_coherence_section=("semantic_coherence_section", "mean"),
            mean_row_underutilization=("row_underutilization_count", "mean"),
            mean_orphan_fields=("orphan_field_count", "mean"),
            mean_section_boundary_misplacement=(
                "section_boundary_misplacement_score",
                "mean",
            ),
            form_level_grid_constraint_violation=(
                "has_grid_constraint_violation",
                "mean",
            ),
            form_level_row_underutilization=("has_row_underutilization", "mean"),
            form_level_orphan_field=("has_orphan_field", "mean"),
            form_level_section_boundary_misplacement=(
                "has_section_boundary_misplacement",
                "mean",
            ),
            form_level_reading_order_violation=(
                "has_reading_order_violation",
                "mean",
            ),
            mean_validation_errors=("validation_error_count", "mean"),
            mean_reading_order_violation_rate=(
                "reading_order_violation_rate",
                "mean",
            ),
        )
        .sort_values("method")
    )


def write_metrics_csv(metrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0].model_dump().keys()))
        writer.writeheader()
        for record in metrics:
            writer.writerow(record.model_dump())


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True))
            file.write("\n")


def to_markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append("| " + " | ".join(format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return cleaned or "default"


if __name__ == "__main__":
    main()
