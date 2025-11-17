"""Document preprocessing helper.

Runs document caching + visual/text index building without Qwen inference.
사용 예시:
    CUDA_VISIBLE_DEVICES=1 python run_preprocessing.py --dataset feta_tab
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from visdomrag import VisDoMRAGConfig, RetrievalManager, load_dataset

logger = logging.getLogger("preprocess")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute caches and indexes")
    parser.add_argument("--dataset", required=True, help="Dataset folder name (e.g., feta_tab)")
    parser.add_argument("--root", default=".", help="Project root (default: current directory)")
    parser.add_argument("--force", action="store_true", help="Force rebuild caches/indexes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    root = Path(args.root).resolve()
    data_dir = root / args.dataset
    output_dir = root / "outputs" / f"{args.dataset}_preprocess"

    config = VisDoMRAGConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        csv_path=None,
        force_reindex=args.force,
    )
    df = load_dataset(config)
    retrieval = RetrievalManager(config=config, df=df)

    logger.info("Caching documents ...")
    retrieval.cache_documents()
    logger.info("Building visual index ...")
    retrieval.build_visual_index()
    logger.info("Building text index ...")
    retrieval.build_text_index()
    logger.info("Preprocessing finished for %s", args.dataset)


if __name__ == "__main__":
    main()
