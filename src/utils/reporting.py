"""Helpers for writing the Sprint 1 reports.

Reports are written in two forms:

* **Markdown** - readable, meant to be shown/printed as project evidence.
* **JSON** - the same numbers in machine-readable form, so later sprints can
  compare against Sprint 1 without re-parsing prose.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars and containers into plain Python types.

    ``json.dump`` cannot serialise ``np.int64`` / ``np.bool_`` etc., which
    appear throughout the pandas-derived summaries.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def save_json(data: dict, path: Path | str) -> Path:
    """Write ``data`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(data), handle, indent=2)
    return path


class MarkdownReport:
    """Small builder for readable Markdown reports.

    Deliberately minimal - just enough structure to keep the report code
    declarative instead of a pile of string concatenation.
    """

    def __init__(self, title: str, subtitle: str | None = None) -> None:
        self.lines: list[str] = [f"# {title}", ""]
        if subtitle:
            self.lines += [f"*{subtitle}*", ""]
        self.lines += [
            f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
            "",
        ]

    def heading(self, text: str, level: int = 2) -> "MarkdownReport":
        self.lines += ["", f"{'#' * level} {text}", ""]
        return self

    def text(self, text: str) -> "MarkdownReport":
        self.lines += [text, ""]
        return self

    def bullets(self, items: list[str]) -> "MarkdownReport":
        self.lines += [f"- {item}" for item in items] + [""]
        return self

    def key_values(self, data: dict, *, headers: tuple[str, str] = ("Field", "Value")) -> "MarkdownReport":
        """Render a flat mapping as a two-column table."""
        self.lines += [
            f"| {headers[0]} | {headers[1]} |",
            "| --- | --- |",
        ]
        for key, value in data.items():
            self.lines.append(f"| {key} | {_format_cell(value)} |")
        self.lines.append("")
        return self

    def table(self, rows: list[dict], columns: list[str] | None = None) -> "MarkdownReport":
        """Render a list of uniform dicts as a Markdown table."""
        if not rows:
            self.lines += ["_(no rows)_", ""]
            return self
        columns = columns or list(rows[0].keys())
        # Column names are not always strings: a pandas crosstab produces
        # integer column labels, which would break the join below.
        headers = [str(c) for c in columns]
        self.lines += [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            self.lines.append(
                "| " + " | ".join(_format_cell(row.get(c, "")) for c in columns) + " |"
            )
        self.lines.append("")
        return self

    def dataframe(self, frame, *, max_rows: int = 20) -> "MarkdownReport":
        """Render (a head of) a pandas DataFrame as a Markdown table."""
        shown = frame.head(max_rows)
        self.table(shown.to_dict("records"), list(shown.columns))
        if len(frame) > max_rows:
            self.text(f"_...{len(frame) - max_rows} more rows_")
        return self

    def code(self, text: str, language: str = "") -> "MarkdownReport":
        self.lines += [f"```{language}", text, "```", ""]
        return self

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path

    def __str__(self) -> str:  # pragma: no cover - convenience for notebooks
        return "\n".join(self.lines)


def _format_cell(value: Any) -> str:
    """Format a single table cell, keeping long containers readable."""
    value = _json_safe(value)
    if isinstance(value, float):
        return f"{value:,.4g}"
    if isinstance(value, dict):
        return ", ".join(f"`{k}`={v}" for k, v in value.items()) or "-"
    if isinstance(value, list):
        if not value:
            return "none"
        text = ", ".join(str(v) for v in value)
        return text if len(text) <= 120 else text[:117] + "..."
    if value is None:
        return "-"
    return str(value)
