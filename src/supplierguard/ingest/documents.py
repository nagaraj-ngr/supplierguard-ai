"""Load supplier documents (.txt or .pdf) from the data directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class Document:
    doc_id: str
    supplier_id: str
    supplier_name: str
    doc_type: str
    text: str


def read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def load_documents(data_dir) -> List[Document]:
    root = Path(data_dir)
    names = {s["id"]: s["name"] for s in json.loads((root / "suppliers.json").read_text(encoding="utf-8"))}
    docs: List[Document] = []
    for supplier_dir in sorted((root / "documents").iterdir()):
        if not supplier_dir.is_dir():
            continue
        sid = supplier_dir.name
        for path in sorted(supplier_dir.iterdir()):
            if path.suffix == ".txt":
                text = path.read_text(encoding="utf-8")
            elif path.suffix == ".pdf":
                text = read_pdf(path)
            else:
                continue
            docs.append(Document(f"{sid}/{path.stem}", sid, names.get(sid, sid), path.stem, text))
    return docs
