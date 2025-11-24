"""경량 Qwen 파이프라인.

전처리된 retrieval CSV/임베딩을 사용해 ColPali/ColQwen을 다시 로드하지 않고
Qwen 추론만 수행합니다.

사용 예시:
    CUDA_VISIBLE_DEVICES=2,3 python run_qwen_light.py --dataset feta_tab
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pdf2image import convert_from_path
from tqdm import tqdm

from visdomrag import (
    VisDoMRAGConfig,
    extract_sections,
    combine_responses,
    generate_textual_response,
    generate_visual_response,
    init_qwen,
    load_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Qwen runner")
    parser.add_argument("--dataset", required=True, help="데이터셋 폴더 이름 (예: feta_tab)")
    parser.add_argument("--root", default=".", help="프로젝트 루트 경로")
    parser.add_argument("--only", default=None, help="쉼표로 구분된 q_id 목록만 처리")
    parser.add_argument("--resume", action="store_true", help="이미 처리된 q_id 건너뛰기")
    parser.add_argument("--top-k", type=int, default=None, help="시각/텍스트 공통 상위 k")
    parser.add_argument(
        "--vision-top-k",
        type=int,
        default=5,
        help="Visual RAG 컨텍스트 개수 (기본값 5)",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=7,
        help="Text RAG 컨텍스트 개수 (기본값 7)",
    )
    parser.add_argument(
        "--vision-retriever",
        choices=["colpali", "colqwen"],
        default="colqwen",
        help="시각 검색기 선택",
    )
    parser.add_argument(
        "--text-retriever",
        choices=["bm25", "minilm", "mpnet", "bge"],
        default="bge",
        help="텍스트 검색기 선택",
    )
    return parser.parse_args()


def safe_query_id(qid: str) -> str:
    return str(qid).replace('/', '$')


def try_parse_answer(answer):
    if isinstance(answer, str):
        try:
            return ast.literal_eval(answer)
        except Exception:
            return answer
    return answer


def load_visual_contexts(config: VisDoMRAGConfig, qid: str, top_k: int) -> List[Dict]:
    csv_path = config.vision_retrieval_file
    if not csv_path.exists():
        raise FileNotFoundError(f"시각 retrieval CSV가 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    rows = df[df['q_id'] == qid]
    if rows.empty:
        return []
    top_rows = rows.nlargest(top_k, 'score')
    contexts: List[Dict] = []
    pdf_dir = config.data_dir / 'docs'
    for _, row in top_rows.iterrows():
        document_id = row['document_id']
        base_doc_id, page_str = document_id.rsplit('_', 1)
        page_idx = int(page_str)
        pdf_path = pdf_dir / f"{base_doc_id}.pdf"
        if not pdf_path.exists():
            continue
        images = convert_from_path(
            str(pdf_path),
            first_page=page_idx + 1,
            last_page=page_idx + 1,
        )
        if not images:
            continue
        contexts.append({
            'image': images[0],
            'document_id': document_id,
            'page_number': page_idx,
        })
    return contexts


def load_textual_contexts(config: VisDoMRAGConfig, qid: str, top_k: int) -> List[Dict]:
    csv_path = config.text_retrieval_file
    if not csv_path.exists():
        raise FileNotFoundError(f"텍스트 retrieval CSV가 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    rows = df[df['q_id'] == qid]
    if rows.empty:
        return []
    top_rows = rows.sort_values('rank').head(top_k)
    contexts = []
    for _, row in top_rows.iterrows():
        contexts.append({
            'chunk': row['chunk'],
            'chunk_pdf_name': row.get('chunk_pdf_name', 'unknown'),
            'pdf_page_number': int(row.get('pdf_page_number', 0)),
        })
    return contexts


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=4)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    data_dir = root / args.dataset
    output_dir = root / 'outputs' / f"{args.dataset}_qwen"

    top_k_candidates = [args.top_k, args.vision_top_k, args.text_top_k]
    base_top_k = max([k for k in top_k_candidates if k], default=5)

    config = VisDoMRAGConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        vision_retriever=args.vision_retriever,
        text_retriever=args.text_retriever,
        top_k=base_top_k,
    )
    config.ensure_directories()
    df = load_dataset(config)
    qwen = init_qwen(device_map='auto')

    if args.only:
        q_ids = [qid.strip() for qid in args.only.split(',') if qid.strip()]
    else:
        q_ids = df['q_id'].astype(str).unique().tolist()

    vision_top_k = args.vision_top_k or args.top_k or config.top_k
    text_top_k = args.text_top_k or args.top_k or config.top_k

    for qid in tqdm(q_ids, desc=f"Qwen ({args.dataset})"):
        row = df[df['q_id'].astype(str) == qid]
        if row.empty:
            continue
        query = row.iloc[0]
        question = str(query['question'])
        answer = try_parse_answer(query.get('answer'))
        safe_qid = safe_query_id(qid)

        visual_file = config.visual_output_dir / f"response_{safe_qid}.json"
        textual_file = config.textual_output_dir / f"response_{safe_qid}.json"
        combined_file = config.combined_output_dir / f"response_{safe_qid}.json"

        if args.resume and combined_file.exists():
            continue

        # Visual
        if visual_file.exists():
            visual_response = json.loads(visual_file.read_text())
        else:
            vctx = load_visual_contexts(config, qid, vision_top_k)
            if not vctx:
                continue
            visual_text = generate_visual_response(qwen, question, [c['image'] for c in vctx], config.qa_prompt)
            visual_response = extract_sections(visual_text)
            visual_response.update({
                'question': question,
                'document': [c['document_id'] for c in vctx],
                'gt_answer': answer,
                'pages': [c['page_number'] for c in vctx],
            })
            save_json(visual_file, visual_response)

        # Textual
        if textual_file.exists():
            textual_response = json.loads(textual_file.read_text())
        else:
            tctx = load_textual_contexts(config, qid, text_top_k)
            if not tctx:
                continue
            textual_text = generate_textual_response(qwen, question, [c['chunk'] for c in tctx], config.qa_prompt)
            textual_response = extract_sections(textual_text)
            textual_response.update({
                'question': question,
                'document': [c['chunk_pdf_name'] for c in tctx],
                'gt_answer': answer,
                'pages': [c['pdf_page_number'] for c in tctx],
                'chunks': "\n".join([c['chunk'] for c in tctx]),
            })
            save_json(textual_file, textual_response)

        combined = combine_responses(qwen, question, visual_response, textual_response)
        combined_payload = {
            'question': question,
            'answer': combined.get('Final Answer', ''),
            'gt_answer': answer,
            'analysis': combined.get('Analysis', ''),
            'conclusion': combined.get('Conclusion', ''),
            'response1': visual_response,
            'response2': textual_response,
        }
        save_json(combined_file, combined_payload)

        gc.collect()
    del qwen
    gc.collect()


if __name__ == '__main__':
    main()
