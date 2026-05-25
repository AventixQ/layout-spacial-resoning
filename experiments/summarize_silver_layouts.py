"""Summarize silver-layout generation and review audit records."""

from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path


def main() -> None:
    audit_path = Path(
        os.environ.get(
            "SILVER_LAYOUT_AUDIT_OUTPUT",
            "outputs/fine_tuning/silver_train_audit.jsonl",
        )
    )
    output_dir = Path(
        os.environ.get(
            "SILVER_LAYOUT_SUMMARY_DIR",
            "outputs/fine_tuning/silver_summary",
        )
    )
    records = _load_jsonl(audit_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    provider_rows = provider_summary(records)
    reviewer_rows = reviewer_summary(records)
    form_rows = form_summary(records)

    _write_csv(provider_rows, output_dir / "provider_summary.csv")
    _write_csv(reviewer_rows, output_dir / "reviewer_summary.csv")
    _write_csv(form_rows, output_dir / "form_summary.csv")
    (output_dir / "provider_summary.md").write_text(
        _markdown_table(provider_rows),
        encoding="utf-8",
    )
    (output_dir / "reviewer_summary.md").write_text(
        _markdown_table(reviewer_rows),
        encoding="utf-8",
    )
    print(f"Wrote silver layout summaries to {output_dir}.")


def provider_summary(records: list[dict]) -> list[dict[str, object]]:
    stats = defaultdict(lambda: {
        "generated": 0,
        "selected_for_repair": 0,
        "final_selected": 0,
        "review_score_sum": 0.0,
        "review_score_count": 0,
        "validation_error_candidates": 0,
        "review_issue_count": 0,
    })
    for record in records:
        for candidate in record.get("candidates", []):
            provider = candidate["author_provider"]
            row = stats[provider]
            row["generated"] += 1
            row["selected_for_repair"] += int(candidate.get("selected_for_repair", False))
            row["final_selected"] += int(candidate.get("final_selected", False))
            row["validation_error_candidates"] += int(bool(candidate.get("validation_errors")))
            row["review_issue_count"] += candidate.get("review_issue_count", 0)
            for score in candidate.get("review_scores", []):
                row["review_score_sum"] += score
                row["review_score_count"] += 1

    rows = []
    for provider, row in sorted(stats.items()):
        generated = row["generated"]
        selected = row["final_selected"]
        rows.append({
            "provider": provider,
            "generated": generated,
            "selected_for_repair": row["selected_for_repair"],
            "final_selected": selected,
            "final_selection_rate": _ratio(selected, generated),
            "mean_review_score": _ratio(row["review_score_sum"], row["review_score_count"]),
            "validation_error_candidates": row["validation_error_candidates"],
            "review_issue_count": row["review_issue_count"],
        })
    return rows


def reviewer_summary(records: list[dict]) -> list[dict[str, object]]:
    stats = defaultdict(lambda: {
        "scores_given": 0,
        "score_sum": 0.0,
        "top_two_votes": 0,
    })
    for record in records:
        for candidate in record.get("candidates", []):
            for reviewer, score in candidate.get("reviewer_scores", {}).items():
                row = stats[reviewer]
                row["scores_given"] += 1
                row["score_sum"] += score
            for reviewer, voted in candidate.get("reviewer_top_two_votes", {}).items():
                if voted:
                    stats[reviewer]["top_two_votes"] += 1

    return [
        {
            "reviewer": reviewer,
            "scores_given": row["scores_given"],
            "mean_score_given": _ratio(row["score_sum"], row["scores_given"]),
            "top_two_votes": row["top_two_votes"],
        }
        for reviewer, row in sorted(stats.items())
    ]


def form_summary(records: list[dict]) -> list[dict[str, object]]:
    rows = []
    for record in records:
        selected_providers = [
            candidate["author_provider"]
            for candidate in record.get("candidates", [])
            if candidate.get("final_selected")
        ]
        rows.append({
            "form_id": record["form_id"],
            "target_layout_count": record.get("target_layout_count", 0),
            "candidate_count": len(record.get("candidates", [])),
            "selected_providers": ",".join(selected_providers),
            "error_count": len(record.get("errors", [])),
            "error_stages": ",".join(
                sorted(Counter(error.get("stage", "unknown") for error in record.get("errors", [])))
            ),
        })
    return rows


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Audit file does not exist: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


if __name__ == "__main__":
    main()
