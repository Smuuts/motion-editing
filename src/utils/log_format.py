"""
Rendering helpers for console output: rules, aligned key/value blocks and tables.

Kept separate from `utils.logger` so the Logger stays about *routing* messages
(console / file / JSONL / W&B) while this module is about *shaping* them. Every
function returns a string and prints nothing, so they are testable in isolation and
usable from any sink.
"""

from __future__ import annotations

import shutil
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_WIDTH = 88
_RULE = "─"


def terminal_width(maximum: int = DEFAULT_WIDTH) -> int:
    """Usable console width, capped so output stays readable on very wide terminals."""
    return min(shutil.get_terminal_size((maximum, 24)).columns, maximum)


def format_value(value: Any, precision: int = 4) -> str:
    """Render a scalar for human reading.

    Floats get `precision` SIGNIFICANT digits, not decimal places, so a table mixing
    losses (1e-4) and percentages (73.6) stays readable without a per-column format.
    Pre-formatted strings pass through untouched, which is how a caller overrides this.
    """
    if isinstance(value, bool) or not isinstance(value, float):
        if isinstance(value, (list, tuple)) and len(value) <= 8:
            return "[" + ", ".join(format_value(v, precision) for v in value) + "]"
        return str(value)
    if value != value or value in (float("inf"), float("-inf")):
        return str(value)
    return f"{value:.{precision}g}"


def rule(title: str | None = None, width: int | None = None, char: str = _RULE) -> str:
    """A horizontal rule, optionally with an inline title: `── title ──────────`."""
    width = width or terminal_width()
    if not title:
        return char * width
    head = f"{char * 2} {title} "
    return head + char * max(width - len(head), 0)


def section(title: str, width: int | None = None) -> str:
    """A blank line plus a titled rule — the standard separator between run phases."""
    return "\n" + rule(title, width)


def key_values(mapping: Mapping[str, Any], indent: int = 2,
               precision: int = 4) -> str:
    """Aligned `key : value` block, one entry per line."""
    items = list(mapping.items())
    if not items:
        return ""
    pad = max(len(str(k)) for k, _ in items)
    lead = " " * indent
    return "\n".join(f"{lead}{str(k):<{pad}} : {format_value(v, precision)}"
                     for k, v in items)


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]], indent: int = 2,
          precision: int = 4, align: str | None = None) -> str:
    """Fixed-width table with a dashed underline beneath the header row.

    `align` is one character per column, "l" or "r"; the default right-aligns every
    column but the first, which is what every metric table in this repo wants.
    """
    body = [[format_value(cell, precision) for cell in row] for row in rows]
    headers = [str(h) for h in headers]
    if not body:
        return f"{' ' * indent}{'  '.join(headers)}\n{' ' * indent}(no rows)"

    n_cols = max(len(headers), max(len(r) for r in body))
    headers = list(headers) + [""] * (n_cols - len(headers))
    body = [list(r) + [""] * (n_cols - len(r)) for r in body]
    align = align or "l" + "r" * (n_cols - 1)
    widths = [max(len(headers[c]), *(len(r[c]) for r in body)) for c in range(n_cols)]

    def render(cells: Sequence[str]) -> str:
        parts = [c.rjust(w) if align[i] == "r" else c.ljust(w)
                 for i, (c, w) in enumerate(zip(cells, widths))]
        return " " * indent + "  ".join(parts).rstrip()

    lines = [render(headers), " " * indent + "  ".join("-" * w for w in widths)]
    lines += [render(r) for r in body]
    return "\n".join(lines)
