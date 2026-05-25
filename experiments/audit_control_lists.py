"""Audit unsorted control lists in the generated form dataset."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset.quality_checks import (
    audit_control_lists_file,
    write_audit_report,
)


def main() -> None:
    input_path = Path(os.environ.get("CONTROL_LIST_AUDIT_INPUT", "data/processed/forms.jsonl"))
    output_dir = Path(os.environ.get("CONTROL_LIST_AUDIT_OUTPUT_DIR", "outputs/dataset_audit"))
    threshold = float(os.environ.get("CONTROL_LIST_SIMILARITY_THRESHOLD", "0.9"))

    report = audit_control_lists_file(
        input_path,
        high_similarity_threshold=threshold,
    )
    write_audit_report(
        report,
        json_path=output_dir / "control_list_audit.json",
        markdown_path=output_dir / "control_list_audit.md",
    )

    print(f"Audited {report.form_count} forms from {input_path}.")
    print(f"Errors: {report.error_count}")
    print(f"Warnings: {report.warning_count}")
    print(f"Similar pairs: {len(report.similar_form_pairs)}")
    print(f"Wrote report to {output_dir / 'control_list_audit.md'}")


if __name__ == "__main__":
    main()
