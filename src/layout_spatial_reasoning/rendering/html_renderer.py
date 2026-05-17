"""Render generated layouts to HTML for qualitative inspection."""

from html import escape

from layout_spatial_reasoning.schemas.control import Control
from layout_spatial_reasoning.schemas.layout import Layout, LayoutControl


def render_html(controls: list[Control], layout: Layout) -> str:
    """Render a generated layout as a compact twelve-column HTML grid."""
    control_by_id = {control.id: control for control in controls}
    sections_html = []
    for section in layout.sections:
        rows_html = []
        for row in section.rows:
            controls_html = [
                _render_control(layout_control, control_by_id.get(layout_control.id))
                for layout_control in row.controls
            ]
            rows_html.append(f'<div class="lsr-row">{"".join(controls_html)}</div>')

        sections_html.append(
            '<section class="lsr-section">'
            f'<h3>{escape(section.section_name)}</h3>'
            f'{"".join(rows_html)}'
            "</section>"
        )

    return f"{_style()}<div class=\"lsr-form\">{''.join(sections_html)}</div>"


def _render_control(layout_control: LayoutControl, control: Control | None) -> str:
    label = control.label if control else f"Unknown control {layout_control.id}"
    help_text = control.help_text if control else ""
    control_type = control.type if control else "text"
    input_html = _input_html(control_type)
    help_html = f'<p class="lsr-help">{escape(help_text)}</p>' if help_text else ""

    return (
        '<div class="lsr-field" '
        f'style="grid-column: {layout_control.colStart} / span {layout_control.colSpan};">'
        f'<label>{escape(label)}</label>'
        f"{input_html}"
        f"{help_html}"
        "</div>"
    )


def _input_html(control_type: str) -> str:
    if control_type == "long_text":
        return "<textarea rows=\"4\"></textarea>"
    if control_type == "boolean":
        return '<div class="lsr-checkbox"><input type="checkbox" /> <span>Yes</span></div>'
    if control_type == "date":
        return '<input type="date" />'
    if control_type == "number":
        return '<input type="number" />'
    if control_type == "file":
        return '<input type="file" />'
    if control_type in {"choice", "multichoice"}:
        multiple = " multiple" if control_type == "multichoice" else ""
        return f"<select{multiple}><option>Option</option></select>"
    return '<input type="text" />'


def _style() -> str:
    return """
<style>
.lsr-form {
  font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172033;
}
.lsr-section {
  border: 1px solid #d7dce5;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  background: #ffffff;
}
.lsr-section h3 {
  margin: 0 0 12px;
  font-size: 16px;
}
.lsr-row {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 12px;
}
.lsr-field {
  min-width: 0;
}
.lsr-field label {
  display: block;
  margin-bottom: 6px;
  font-size: 13px;
  font-weight: 650;
}
.lsr-field input,
.lsr-field textarea,
.lsr-field select {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid #c9d1dd;
  border-radius: 6px;
  padding: 8px 10px;
  background: #ffffff;
}
.lsr-checkbox {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 38px;
}
.lsr-checkbox input {
  width: auto;
}
.lsr-help {
  margin: 5px 0 0;
  font-size: 12px;
  color: #667085;
}
</style>
"""
