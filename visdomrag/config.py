from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

import pandas as pd

DEFAULT_QA_PROMPT = "Answer the question objectively based on the provided context."


@dataclass
class VisDoMRAGConfig:
    """노트북과 모듈에서 공동으로 참조하는 핵심 설정 값을 묶어둔 컨테이너."""

    data_dir: Path
    output_dir: Path
    csv_path: Path | None = None
    llm_model: str = "qwen"
    vision_retriever: str = "colpali"
    text_retriever: str = "bm25"
    top_k: int = 5
    chunk_size: int = 3000
    chunk_overlap: int = 300
    qa_prompt: str = DEFAULT_QA_PROMPT
    force_reindex: bool = False
    api_keys: Dict[str, Any] = field(default_factory=dict)
    vision_device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    vision_torch_dtype: str | None = None  # e.g. "bfloat16", "float16", "float32"

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.csv_path:
            self.csv_path = Path(self.csv_path)

    @property
    def dataset_csv(self) -> Path:
        return self.csv_path or self.data_dir / f"{self.data_dir.name}.csv"

    @property
    def retrieval_dir(self) -> Path:
        return self.data_dir / "retrieval"

    @property
    def vision_retrieval_file(self) -> Path:
        return self.retrieval_dir / f"retrieval_{self.vision_retriever}.csv"

    @property
    def text_retrieval_file(self) -> Path:
        return self.retrieval_dir / f"retrieval_{self.text_retriever}.csv"

    @property
    def visual_output_dir(self) -> Path:
        return self.output_dir / f"{self.llm_model}_vision"

    @property
    def textual_output_dir(self) -> Path:
        return self.output_dir / f"{self.llm_model}_text"

    @property
    def combined_output_dir(self) -> Path:
        return self.output_dir / f"{self.llm_model}_visdmrag"

    def ensure_directories(self) -> None:
        self.visual_output_dir.mkdir(parents=True, exist_ok=True)
        self.textual_output_dir.mkdir(parents=True, exist_ok=True)
        self.combined_output_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_dir.mkdir(parents=True, exist_ok=True)


def load_dataset(config: VisDoMRAGConfig) -> pd.DataFrame:
    """설정 정보에 명시된 CSV 파일을 판다스 데이터프레임으로 불러온다."""
    csv_path = config.dataset_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


__all__ = ["VisDoMRAGConfig", "DEFAULT_QA_PROMPT", "load_dataset"]
