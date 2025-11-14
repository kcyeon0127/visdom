"""Batch runner for VisDoMRAG (Qwen-only).

Usage:
    python run_all_datasets.py

This iterates over predefined datasets, runs the pipeline end-to-end,
then evaluates the combined responses (word-level F1) and saves a summary.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from visdomrag import (
    VisDoMRAGConfig,
    load_dataset,
    RetrievalManager,
    init_qwen,
    run_pipeline,
)

import re
import string
from collections import Counter

logger = logging.getLogger("run_all")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def word_tokenize(text: str) -> List[str]:
    return normalize_answer(text).split()


def calculate_f1(pred: List[str], gold: List[str]) -> float:
    pc = Counter(pred)
    gc = Counter(gold)
    tp = sum((pc & gc).values())
    fp = sum(pc.values()) - tp
    fn = sum(gc.values()) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def evaluate_json_dir(path: Path) -> Dict:
    records = []
    for json_file in sorted(path.glob("*.json")):
        with json_file.open() as fh:
            data = json.load(fh)
        pred = str(data.get("Answer") or data.get("answer") or "").strip()
        gold = str(data.get("gt_answer") or "").strip()
        if not pred or not gold:
            score = None
        else:
            score = calculate_f1(word_tokenize(pred), word_tokenize(gold))
        records.append({"file": json_file.name, "f1": score})

    valid_scores = [r["f1"] for r in records if r["f1"] is not None]
    avg = mean(valid_scores) if valid_scores else None
    return {
        "records": records,
        "average_f1": avg,
        "evaluated_files": len(valid_scores),
        "total_files": len(records),
    }


def run_for_dataset(name: str, data_dir: Path, output_root: Path, csv_path: Optional[Path]) -> Dict:
    logger.info("Running dataset %s", name)
    config = VisDoMRAGConfig(
        data_dir=data_dir,
        output_dir=output_root / name,
        csv_path=csv_path,
    )
    df = load_dataset(config)
    retrieval = RetrievalManager(config=config, df=df)
    run_pipeline(config, retrieval, qwen_model)
    eval_dir = config.combined_output_dir
    stats = evaluate_json_dir(eval_dir)
    stats.update({"dataset": name, "eval_dir": str(eval_dir)})
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path.cwd()

    datasets = {
        "feta_tab": {
            "data_dir": root / "feta_tab",
            "csv": None,
        },
        # Add other datasets here
    }

    qwen_model = init_qwen()
    output_root = root / "outputs"

    summary: List[Dict] = []
    for name, cfg in datasets.items():
        stats = run_for_dataset(name, cfg["data_dir"], output_root, cfg.get("csv"))
        summary.append(stats)

    summary_path = root / "run_all_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved summary to %s", summary_path)
