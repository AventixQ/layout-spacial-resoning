"""Run Method 1 (single-LLM) on the validation split with a cheap model.

Default model: gpt-4o-mini (OpenAI).
Override via env vars:
  LLM_PROVIDER=gemini   LAYOUT_MODEL=gemini-2.5-flash
  LLM_PROVIDER=claude   LAYOUT_MODEL=claude-haiku-4-5

Writes results to outputs/experiments/validation/<provider>_<model>/
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.config import load_env
from layout_spatial_reasoning.dataset import load_forms_jsonl, write_layouts_jsonl
from layout_spatial_reasoning.methods.llm_single import PromptVariant, generate_layout

load_env()

PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai")
MODEL: str = os.environ.get("LAYOUT_MODEL", "gpt-4o-mini")

VARIANTS: list[PromptVariant] = ["zero_shot", "cot", "structured_output"]

INPUT_PATH = Path("data/splits/val.jsonl")
OUTPUT_DIR = Path("outputs/experiments/validation") / f"{PROVIDER}_{MODEL.replace('/', '_')}"


def main() -> None:
    forms = load_forms_jsonl(INPUT_PATH)
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {MODEL}")
    print(f"Forms    : {len(forms)}")
    print(f"Variants : {', '.join(VARIANTS)}")
    print(f"Total API calls: {len(forms) * len(VARIANTS)}")
    print(f"Output   : {OUTPUT_DIR}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records: list[tuple[str, str, object]] = []
    errors: list[dict] = []
    t0 = time.time()

    for i, form in enumerate(forms, 1):
        for variant in VARIANTS:
            method = f"llm_single_{variant}"
            try:
                layout = generate_layout(
                    form.controls,
                    variant=variant,
                    provider=PROVIDER,
                    model=MODEL,
                )
                records.append((form.form_id, method, layout))
            except Exception as exc:  # noqa: BLE001
                errors.append({
                    "form_id": form.form_id,
                    "method": method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

        if i % 20 == 0:
            elapsed = time.time() - t0
            remaining = (len(forms) - i) / i * elapsed
            print(
                f"  [{i:>3}/{len(forms)}]  ok={len(records)}  err={len(errors)}"
                f"  elapsed={elapsed:.0f}s  ~{remaining:.0f}s left"
            )

    write_layouts_jsonl(records, OUTPUT_DIR / "layouts.jsonl")
    _write_errors(errors, OUTPUT_DIR / "errors.jsonl")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Layouts : {len(records)}  →  {OUTPUT_DIR / 'layouts.jsonl'}")
    print(f"  Errors  : {len(errors)}  →  {OUTPUT_DIR / 'errors.jsonl'}")
    if errors:
        by_type: dict[str, int] = {}
        for e in errors:
            by_type[e["error_type"]] = by_type.get(e["error_type"], 0) + 1
        for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count}")


def _write_errors(errors: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=True))
            f.write("\n")


if __name__ == "__main__":
    main()
