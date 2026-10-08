"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(
    os.environ.get("SUPPLIERGUARD_ROOT") or Path(__file__).resolve().parents[2]
)

EMBEDDER_KINDS = ("auto", "openai", "hash")


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str]
    llm_model: str
    embedding_model: str
    embedder_kind: str
    seed: int
    data_dir: Path
    index_dir: Path

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)


def load_settings(env: Mapping[str, str]) -> Settings:
    """Build Settings from a mapping. Pure function, no file or global access."""
    kind = env.get("SUPPLIERGUARD_EMBEDDER", "auto").strip().lower() or "auto"
    if kind not in EMBEDDER_KINDS:
        raise ValueError(f"SUPPLIERGUARD_EMBEDDER must be one of {EMBEDDER_KINDS}, got {kind!r}")
    data_dir = Path(env.get("SUPPLIERGUARD_DATA_DIR") or PROJECT_ROOT / "data")
    return Settings(
        openai_api_key=(env.get("OPENAI_API_KEY") or "").strip() or None,
        llm_model=env.get("SUPPLIERGUARD_LLM_MODEL", "gpt-4o-mini"),
        embedding_model=env.get("SUPPLIERGUARD_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedder_kind=kind,
        seed=int(env.get("SUPPLIERGUARD_SEED", "42")),
        data_dir=data_dir,
        index_dir=Path(env.get("SUPPLIERGUARD_INDEX_DIR") or data_dir / "index"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # python-dotenv is optional at runtime
        pass
    return load_settings(os.environ)
