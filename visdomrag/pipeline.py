from __future__ import annotations

import ast
import gc
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from .config import VisDoMRAGConfig
from .qwen import (
    QwenResources,
    generate_text_only_response,
    generate_textual_response,
    generate_visual_response,
)
from .retrieval import RetrievalManager

logger = logging.getLogger(__name__)


def _safe_query_id(query_id: str) -> str:
    """파일명에 쓸 수 있도록 쿼리 ID에서 / 등을 안전한 문자로 치환."""
    return str(query_id).replace('/', '$')


def _try_parse_answer(raw_answer) -> object:
    """문자열로 저장된 정답 필드를 literal 변환해 원래 타입에 가깝게 복원."""
    if isinstance(raw_answer, str):
        try:
            return ast.literal_eval(raw_answer)
        except Exception:
            return raw_answer
    return raw_answer


def extract_sections(text: str) -> Dict[str, str]:
    """Evidence/Chain of Thought/Answer 섹션을 정규식으로 추출."""
    headings = ["Evidence", "Chain of Thought", "Answer"]
    sections: Dict[str, str] = {}
    for idx, heading in enumerate(headings):
        next_heading = headings[idx + 1] if idx + 1 < len(headings) else None
        if next_heading:
            pattern = f"## {heading}:(.*?)(?=## {next_heading}:)"
        else:
            pattern = f"## {heading}:(.*)"
        match = re.search(pattern, text, re.DOTALL)
        sections[heading] = match.group(1).strip() if match else ""
    return sections


def parse_combined_output(text: str) -> Dict[str, str]:
    """응답 결합 단계에서 생성된 ## 헤더 기반 텍스트를 dict로 정리."""
    sections = {'Analysis': '', 'Conclusion': '', 'Final Answer': ''}
    current_section: Optional[str] = None
    for line in text.split('\n'):
        if line.startswith('## '):
            current_section = line[3:].strip(':')
            continue
        if current_section in sections:
            sections[current_section] += line + '\n'
    for key in sections:
        sections[key] = sections[key].strip()
    return sections


def _build_combination_prompt(
    query: str,
    visual_response: Dict[str, str],
    textual_response: Dict[str, str],
) -> str:
    """시각·텍스트 응답을 비교 분석하도록 Qwen에 지시하는 프롬프트 템플릿 구성."""
    return f"""
Analyze the following two responses to the question: \"{query}\"

Response 1:
Evidence: {visual_response.get('Evidence', 'N/A')}
Chain of Thought: {visual_response.get('Chain of Thought', 'N/A')}
Final Answer: {visual_response.get('Answer', 'N/A')}

Response 2:
Evidence: {textual_response.get('Evidence', 'N/A')}
Chain of Thought: {textual_response.get('Chain of Thought', 'N/A')}
Final Answer: {textual_response.get('Answer', 'N/A')}

Response 1 is based on a visual pipeline, and Response 2 on a textual pipeline.
- If both are logical, trust the evidence in Response 1 more.
- If one response refuses to answer, give more weight to the other unless both have strong reasons not to answer.
- Keep the final answer short and direct, mirroring the dataset's style.

Provide your analysis in the following sections:
## Analysis:
[Discuss consistency of evidence, reasoning, and answers.]

## Conclusion:
[Decide which answer (or synthesis) is most reliable.]

## Final Answer:
[Answer the question concisely.]
"""


def combine_responses(
    qwen: QwenResources,
    query: str,
    visual_response: Dict[str, str],
    textual_response: Dict[str, str],
) -> Dict[str, str]:
    """두 파이프라인 출력물을 Qwen으로 재평가해 최종 분석/답변을 얻는다."""
    prompt = _build_combination_prompt(query, visual_response, textual_response)
    output = generate_text_only_response(qwen, prompt, max_new_tokens=1000)
    return parse_combined_output(output)


def _write_json(path: Path, payload: Dict) -> None:
    """출력 디렉터리를 보장한 뒤 JSON 파일로 결과를 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=4, ensure_ascii=False)


def process_query(
    config: VisDoMRAGConfig,
    retrieval: RetrievalManager,
    qwen: QwenResources,
    query_id: str,
) -> bool:
    """단일 쿼리에 대해 retrieval→Qwen 응답→결합→저장을 순차 수행."""
    rows = retrieval.df[retrieval.df['q_id'] == query_id]
    if rows.empty:
        logger.warning("Query %s not found in dataset", query_id)
        return False
    entry = rows.iloc[0]
    question = entry['question']
    answer = _try_parse_answer(entry.get('answer'))
    safe_query = _safe_query_id(query_id)

    visual_file = config.visual_output_dir / f"response_{safe_query}.json"
    textual_file = config.textual_output_dir / f"response_{safe_query}.json"
    combined_file = config.combined_output_dir / f"response_{safe_query}.json"

    if combined_file.exists():
        logger.info("Combined response already exists for %s", query_id)
        return True

    visual_response_dict: Optional[Dict] = None
    textual_response_dict: Optional[Dict] = None

    if visual_file.exists():
        with open(visual_file) as fh:
            visual_response_dict = json.load(fh)
    else:
        visual_contexts = retrieval.retrieve_visual_contexts(query_id)
        if visual_contexts:
            images = [ctx['image'] for ctx in visual_contexts]
            visual_response = generate_visual_response(
                qwen, question, images, config.qa_prompt
            )
            visual_response_dict = extract_sections(visual_response)
            visual_response_dict.update(
                {
                    "question": question,
                    "document": [ctx['document_id'] for ctx in visual_contexts],
                    "gt_answer": answer,
                    "pages": [ctx['page_number'] for ctx in visual_contexts],
                }
            )
            _write_json(visual_file, visual_response_dict)

    if textual_file.exists():
        with open(textual_file) as fh:
            textual_response_dict = json.load(fh)
    else:
        textual_contexts = retrieval.retrieve_textual_contexts(query_id)
        if textual_contexts:
            chunks = [ctx['chunk'] for ctx in textual_contexts]
            textual_response = generate_textual_response(
                qwen, question, chunks, config.qa_prompt
            )
            textual_response_dict = extract_sections(textual_response)
            textual_response_dict.update(
                {
                    "question": question,
                    "document": [ctx['chunk_pdf_name'] for ctx in textual_contexts],
                    "gt_answer": answer,
                    "pages": [ctx['pdf_page_number'] for ctx in textual_contexts],
                    "chunks": "\n".join(chunks),
                }
            )
            _write_json(textual_file, textual_response_dict)

    if not visual_response_dict or not textual_response_dict:
        logger.warning("Missing responses for %s", query_id)
        return False

    combined_sections = combine_responses(
        qwen, question, visual_response_dict, textual_response_dict
    )
    combined_response = {
        "question": question,
        "answer": combined_sections.get("Final Answer", ""),
        "gt_answer": answer,
        "analysis": combined_sections.get("Analysis", ""),
        "conclusion": combined_sections.get("Conclusion", ""),
        "response1": visual_response_dict,
        "response2": textual_response_dict,
    }
    _write_json(combined_file, combined_response)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def run_pipeline(
    config: VisDoMRAGConfig,
    retrieval: RetrievalManager,
    qwen: QwenResources,
    query_ids: Optional[Sequence[str]] = None,
    delay_seconds: float = 1.0,
) -> List[Tuple[str, bool]]:
    """여러 쿼리를 순회하며 `process_query`를 호출하고 성공 여부를 기록."""
    ids = query_ids or list(retrieval.df['q_id'].unique())
    results: List[Tuple[str, bool]] = []
    for query_id in tqdm(ids, desc="Processing queries"):
        try:
            success = process_query(config, retrieval, qwen, query_id)
        except Exception:
            logger.exception("Error while processing %s", query_id)
            success = False
        results.append((query_id, success))
        if delay_seconds:
            time.sleep(delay_seconds)
    return results


__all__ = [
    "process_query",
    "run_pipeline",
    "extract_sections",
    "combine_responses",
    "parse_combined_output",
]
