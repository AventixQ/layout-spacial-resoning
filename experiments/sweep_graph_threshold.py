"""Sweep graph similarity thresholds for Method 4."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from layout_spatial_reasoning.dataset import load_forms_jsonl
from layout_spatial_reasoning.embeddings.provider import embedding_function_from_env
from layout_spatial_reasoning.evaluation.metrics import evaluate_generated_layout
from layout_spatial_reasoning.methods.graph_community import generate_layout
from layout_spatial_reasoning.schemas.results import GeneratedLayoutRecord


THRESHOLDS = [round(value / 100, 2) for value in range(15, 81, 5)]


def main() -> None:
    forms_path = Path("data/processed/sample_forms.jsonl")
    output_path = Path("outputs/metrics/graph_threshold_sweep.csv")
    summary_path = Path("outputs/metrics/graph_threshold_sweep_summary.csv")
    markdown_path = Path("outputs/metrics/graph_threshold_sweep_summary.md")

    embedding_function = _cached_embedding_function(embedding_function_from_env())
    forms = load_forms_jsonl(forms_path)

    rows = []
    for threshold in THRESHOLDS:
        for form in forms:
            layout = generate_layout(
                form.controls,
                embedding_function=embedding_function,
                similarity_threshold=threshold,
                community_algorithm="leiden",
            )
            generated = GeneratedLayoutRecord(
                form_id=form.form_id,
                method=f"graph_community_t{threshold:.2f}",
                layout=layout,
            )
            record = evaluate_generated_layout(
                form,
                generated,
                embedding_function=embedding_function,
            )
            row = record.model_dump()
            row["threshold"] = threshold
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(pd.DataFrame(rows))
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(_to_markdown_table(summary.head(10)), encoding="utf-8")

    best = summary.iloc[0]
    print(f"Wrote sweep rows to {output_path}.")
    print(f"Wrote threshold summary to {summary_path} and {markdown_path}.")
    print(
        "Best threshold: "
        f"{best['threshold']:.2f} "
        f"(orphan={best['mean_orphan_fields']:.4f}, "
        f"section={best['mean_semantic_coherence_section']:.4f}, "
        f"grid={best['mean_grid_utilization']:.4f})"
    )


def _cached_embedding_function(embedding_function):
    cache: dict[tuple[str, ...], list[list[float]]] = {}

    def cached(texts: list[str]) -> list[list[float]]:
        key = tuple(texts)
        if key not in cache:
            cache[key] = embedding_function(texts)
        return cache[key]

    return cached


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby("threshold", as_index=False)
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
            mean_reading_order_violation_rate=(
                "reading_order_violation_rate",
                "mean",
            ),
        )
    )
    return summary.sort_values(
        [
            "mean_orphan_fields",
            "mean_section_boundary_misplacement",
            "mean_row_underutilization",
            "mean_reading_order_violation_rate",
            "mean_semantic_coherence_section",
            "mean_semantic_coherence_row",
            "mean_grid_utilization",
        ],
        ascending=[True, True, True, True, False, False, False],
    )


def _to_markdown_table(summary: pd.DataFrame) -> str:
    columns = list(summary.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        values = [_format_value(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
