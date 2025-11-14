from __future__ import annotations

import importlib
from typing import Dict

__all__ = [
    "DEFAULT_QA_PROMPT",
    "VisDoMRAGConfig",
    "load_dataset",
    "QwenResources",
    "init_qwen",
    "generate_visual_response",
    "generate_textual_response",
    "generate_text_only_response",
    "RetrievalManager",
    "process_query",
    "run_pipeline",
    "extract_sections",
    "combine_responses",
]

_LAZY_IMPORTS: Dict[str, str] = {
    "DEFAULT_QA_PROMPT": "visdomrag.config",
    "VisDoMRAGConfig": "visdomrag.config",
    "load_dataset": "visdomrag.config",
    "QwenResources": "visdomrag.qwen",
    "init_qwen": "visdomrag.qwen",
    "generate_visual_response": "visdomrag.qwen",
    "generate_textual_response": "visdomrag.qwen",
    "generate_text_only_response": "visdomrag.qwen",
    "RetrievalManager": "visdomrag.retrieval",
    "process_query": "visdomrag.pipeline",
    "run_pipeline": "visdomrag.pipeline",
    "extract_sections": "visdomrag.pipeline",
    "combine_responses": "visdomrag.pipeline",
}


def __getattr__(name: str):  # pragma: no cover - import side effect
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'visdomrag' has no attribute '{name}'")
    module = importlib.import_module(_LAZY_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():  # pragma: no cover - tooling helper
    return sorted(__all__)
