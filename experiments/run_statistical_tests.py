"""Run the thesis statistical analysis plan on computed metrics."""

import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from layout_spatial_reasoning.evaluation.statistical_tests import run_statistical_tests


def main() -> None:
    metrics_path = Path(os.environ.get("METRICS_INPUT_PATH", "outputs/metrics/sample_metrics.csv"))
    output_path = Path(
        os.environ.get(
            "STATISTICAL_TESTS_OUTPUT_PATH",
            "outputs/statistical_tests/sample_statistical_tests.json",
        )
    )

    metrics = pd.read_csv(metrics_path)
    results = run_statistical_tests(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote statistical test results to {output_path}.")


if __name__ == "__main__":
    main()
