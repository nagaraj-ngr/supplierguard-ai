"""Wiring shared by the CLI, the evaluation and the UI: settings -> index -> sources -> approach."""

from __future__ import annotations

from typing import Optional

from .approaches import APPROACHES, Assessment
from .config import Settings, get_settings
from .ingest.embeddings import get_embedder
from .ingest.pipeline import build_index
from .ingest.retriever import Retriever
from .llm import ChatModel, get_chat_model
from .sources import SupplierSources


def load_retriever(settings: Settings) -> Retriever:
    embedder = get_embedder(settings.embedder_kind, settings.embedding_model, settings.openai_api_key)
    if not (settings.index_dir / "meta.json").exists():
        build_index(settings.data_dir, settings.index_dir, embedder=embedder)
    return Retriever.load(settings.index_dir, embedder)


def load_sources(settings: Optional[Settings] = None) -> SupplierSources:
    settings = settings or get_settings()
    return SupplierSources(settings.data_dir, retriever=load_retriever(settings))


def assess_supplier(approach: str, supplier_query: str, sources: Optional[SupplierSources] = None,
                    model: Optional[ChatModel] = None) -> Assessment:
    """Resolve the name first, so bad input is rejected before any model is needed or called."""
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach {approach!r}; choose one of {', '.join(APPROACHES)}")
    sources = sources or load_sources()
    supplier_id = sources.resolve(supplier_query)
    model = model or default_model()
    return APPROACHES[approach](supplier_id, sources, model)


def default_model(settings: Optional[Settings] = None) -> ChatModel:
    return get_chat_model(settings or get_settings())
