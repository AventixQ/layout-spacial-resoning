"""Render silver target layouts to a static HTML gallery for inspection."""

from html import escape
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_spatial_reasoning.dataset.io import load_forms_jsonl
from layout_spatial_reasoning.rendering.html_renderer import render_html
from layout_spatial_reasoning.schemas.form import FormSpec


def main() -> None:
    input_path = Path(
        os.environ.get(
            "SILVER_GALLERY_INPUT",
            "outputs/fine_tuning/silver_train_forms_recovered.jsonl",
        )
    )
    output_path = Path(
        os.environ.get(
            "SILVER_GALLERY_OUTPUT",
            "outputs/fine_tuning/silver_layout_gallery.html",
        )
    )
    limit = _optional_int(os.environ.get("SILVER_GALLERY_LIMIT", "80"))
    start = int(os.environ.get("SILVER_GALLERY_START", "0"))

    forms = load_forms_jsonl(input_path)
    selected = forms[start : None if limit is None else start + limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_page(selected, input_path, start, limit), encoding="utf-8")
    print(f"Wrote {len(selected)} rendered forms to {output_path}.")


def _page(
    forms: list[FormSpec],
    input_path: Path,
    start: int,
    limit: int | None,
) -> str:
    cards = "\n".join(_form_card(form) for form in forms)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Silver Layout Gallery</title>
  <style>
    body {{
      margin: 0;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f9;
      color: #172033;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 16px 24px;
      border-bottom: 1px solid #d7dce5;
      background: rgba(255, 255, 255, 0.96);
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
    }}
    .meta {{
      color: #667085;
      font-size: 13px;
    }}
    main {{
      padding: 20px 24px 40px;
    }}
    details.form-card {{
      margin-bottom: 20px;
      border: 1px solid #d7dce5;
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }}
    details.form-card > summary {{
      cursor: pointer;
      padding: 14px 16px;
      font-weight: 700;
      border-bottom: 1px solid #e6eaf0;
    }}
    .form-body {{
      padding: 16px;
    }}
    .controls {{
      margin: 0 0 16px;
      padding: 10px 12px;
      border: 1px solid #e2e7ef;
      border-radius: 6px;
      background: #f8fafc;
      font-size: 12px;
      line-height: 1.5;
    }}
    .layouts {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .layout-panel {{
      min-width: 0;
      border: 1px solid #d7dce5;
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .layout-panel h2 {{
      margin: 0 0 12px;
      font-size: 14px;
    }}
    @media (max-width: 1100px) {{
      .layouts {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Silver Layout Gallery</h1>
    <div class="meta">
      Source: {escape(str(input_path))} · start={start} · limit={limit if limit is not None else "all"} · rendered={len(forms)}
    </div>
  </header>
  <main>
    {cards}
  </main>
</body>
</html>
"""


def _form_card(form: FormSpec) -> str:
    controls = " · ".join(
        f"{escape(control.id)}: {escape(control.label)} ({escape(control.type)})"
        for control in form.controls
    )
    layouts = "\n".join(
        '<div class="layout-panel">'
        f"<h2>Target layout {index}</h2>"
        f"{render_html(form.controls, layout)}"
        "</div>"
        for index, layout in enumerate(form.target_layouts, start=1)
    )
    return f"""
<details class="form-card">
  <summary>{escape(form.form_id)} · {escape(form.domain)} · controls={len(form.controls)} · targets={len(form.target_layouts)}</summary>
  <div class="form-body">
    <div class="controls">{controls}</div>
    <div class="layouts">{layouts}</div>
  </div>
</details>
"""


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    main()
