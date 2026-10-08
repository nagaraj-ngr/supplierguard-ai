"""Optional PDF rendering for generated documents (requires reportlab)."""

from __future__ import annotations

import textwrap
from pathlib import Path


def write_text_pdf(text: str, path: Path, width: int = 92) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    page_w, page_h = letter
    margin, line_h = 54, 11
    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    c.setFont("Courier", 9)
    y = page_h - margin
    for raw in text.splitlines():
        for line in textwrap.wrap(raw, width=width, subsequent_indent="  ") or [""]:
            if y < margin:
                c.showPage()
                c.setFont("Courier", 9)
                y = page_h - margin
            c.drawString(margin, y, line)
            y -= line_h
    c.save()
