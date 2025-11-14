import json
import logging
import sys
from pathlib import Path

from visdomrag import VisDoMRAGConfig, load_dataset, RetrievalManager, init_qwen, run_pipeline

logging.basicConfig(level=logging.INFO)

config = VisDoMRAGConfig(
    data_dir=Path('feta_tab'),
    output_dir=Path('outputs/feta_tab_qwen'),
)

df = load_dataset(config)
retrieval = RetrievalManager(config=config, df=df)
qwen = init_qwen()

run_pipeline(config, retrieval, qwen)
