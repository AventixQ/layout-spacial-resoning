"""Generate the Form Layout Benchmark dataset and split it into train/val/test.

Output:
  data/processed/forms.jsonl   – all 1 600 forms (200 per domain)
  data/splits/train.jsonl      – 1 120 forms (140 per domain)
  data/splits/val.jsonl        –   240 forms ( 30 per domain)
  data/splits/test.jsonl       –   240 forms ( 30 per domain)
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset.build_dataset import build_dataset
from layout_spatial_reasoning.dataset.split_dataset import split_dataset


PROCESSED = Path("data/processed/forms.jsonl")
SPLITS_DIR = Path("data/splits")


def main() -> None:
    print("── Step 1: generating forms ──────────────────────────────")
    forms = build_dataset(PROCESSED, seed=42)

    domains = {}
    for f in forms:
        domains[f.domain] = domains.get(f.domain, 0) + 1
    for domain, count in sorted(domains.items()):
        print(f"  {domain:<30} {count:>4} forms")
    print(f"  {'TOTAL':<30} {len(forms):>4} forms")

    print("\n── Step 2: splitting ─────────────────────────────────────")
    split_dataset(PROCESSED, SPLITS_DIR, seed=42)


if __name__ == "__main__":
    main()
