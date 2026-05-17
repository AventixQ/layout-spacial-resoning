"""Generate tables for the thesis results chapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd


def main() -> None:
    metrics_path = Path("outputs/metrics/sample_metrics.csv")
    output_csv_path = Path("outputs/metrics/sample_method_summary.csv")
    output_markdown_path = Path("outputs/metrics/sample_method_summary.md")

    metrics = pd.read_csv(metrics_path)
    summary = (
        metrics.groupby("method", as_index=False)
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

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv_path, index=False)
    output_markdown_path.write_text(_to_markdown_table(summary), encoding="utf-8")

    print(f"Wrote method summary to {output_csv_path} and {output_markdown_path}.")


def _to_markdown_table(summary: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional pandas dependencies."""
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
