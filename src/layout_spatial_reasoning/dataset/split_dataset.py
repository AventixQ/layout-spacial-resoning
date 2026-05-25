"""Create train, validation, and test splits from the FLB dataset.

Split is performed per domain to maintain balanced class distribution:
  140 train / 30 validation / 30 test per domain  (70 / 15 / 15 %)
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from layout_spatial_reasoning.dataset.io import load_forms_jsonl, write_layouts_jsonl
from layout_spatial_reasoning.schemas.form import FormSpec


TRAIN_PER_DOMAIN = 140
VAL_PER_DOMAIN = 30
TEST_PER_DOMAIN = 30
TOTAL_PER_DOMAIN = TRAIN_PER_DOMAIN + VAL_PER_DOMAIN + TEST_PER_DOMAIN  # 200


def split_dataset(
    input_path: str | Path,
    splits_dir: str | Path,
    seed: int = 42,
) -> dict[str, list[FormSpec]]:
    """Split the FLB dataset by domain and write three JSONL files.

    Returns a dict with keys ``train``, ``val``, ``test`` containing the
    split FormSpec lists.
    """
    forms = load_forms_jsonl(input_path)

    by_domain: dict[str, list[FormSpec]] = defaultdict(list)
    for form in forms:
        by_domain[form.domain].append(form)

    train_forms: list[FormSpec] = []
    val_forms: list[FormSpec] = []
    test_forms: list[FormSpec] = []

    rng = random.Random(seed)
    for domain, domain_forms in sorted(by_domain.items()):
        if len(domain_forms) < TOTAL_PER_DOMAIN:
            raise ValueError(
                f"Domain '{domain}' has {len(domain_forms)} forms, "
                f"need at least {TOTAL_PER_DOMAIN}."
            )
        shuffled = list(domain_forms)
        rng.shuffle(shuffled)
        train_forms.extend(shuffled[:TRAIN_PER_DOMAIN])
        val_forms.extend(shuffled[TRAIN_PER_DOMAIN:TRAIN_PER_DOMAIN + VAL_PER_DOMAIN])
        test_forms.extend(shuffled[TRAIN_PER_DOMAIN + VAL_PER_DOMAIN:TOTAL_PER_DOMAIN])

    out = Path(splits_dir)
    out.mkdir(parents=True, exist_ok=True)

    _write_forms(train_forms, out / "train.jsonl")
    _write_forms(val_forms, out / "val.jsonl")
    _write_forms(test_forms, out / "test.jsonl")

    print(
        f"Split complete → train: {len(train_forms)}, "
        f"val: {len(val_forms)}, test: {len(test_forms)}"
    )
    return {"train": train_forms, "val": val_forms, "test": test_forms}


def _write_forms(forms: list[FormSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for form in forms:
            f.write(form.model_dump_json())
            f.write("\n")
