from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from PIL import Image

import logging

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QwenResources:
    """Qwen2-VL 모델과 프로세서를 보관하는 경량 리소스 래퍼."""

    model: "Qwen2VLForConditionalGeneration"
    processor: "AutoProcessor"
    process_vision_info: callable
    device: str


def init_qwen(
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map: str | None = "auto",
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 640 * 28 * 28,
) -> QwenResources:
    """Qwen2-VL 모델과 프로세서를 메모리에 로드해 inference 자원을 준비한다."""
    try:
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Install transformers and qwen-vl utilities to run Qwen2-VL inference."
        ) from exc

    target_device = "cuda"
    logger.info("Loading Qwen model %s", model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return QwenResources(
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        device=target_device,
    )


def _decode_generation(resources: QwenResources, inputs, generated_ids) -> str:
    """생성 결과에서 프롬프트 토큰을 제거하고 순수 출력 텍스트만 복원."""
    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = resources.processor.batch_decode(
        trimmed, skip_special_tokens=True
    )
    return output_text[0]


def generate_visual_response(
    resources: QwenResources,
    query: str,
    images: Sequence[Image.Image],
    qa_prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """PDF 페이지 이미지를 조건으로 삼아 답변을 생성한다."""
    prompt = (
        "You are tasked with answering a question based on the relevant pages of a PDF document. "
        "Provide your response in the following format:\n"
        "## Evidence:\n\n"
        "## Chain of Thought:\n\n"
        "## Answer:\n\n"
        "___\n"
        "Instructions:\n\n"
        "1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided pages and populate them in the \"Evidence\" section. "
        "For each element, include the type, content, and a brief explanation of its relevance.\n\n"
        "2. Chain of Thought: In the \"Chain of Thought\" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. "
        "You should perform computations if needed.\n\n"
        f"3. Answer: {qa_prompt}\n___\n"
        f"Question: {query}"
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in images]
            + [{"type": "text", "text": prompt}],
        }
    ]

    text = resources.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = resources.process_vision_info(messages)
    inputs = resources.processor(
        text=[text], images=image_inputs, padding=True, return_tensors="pt"
    )
    device = torch.device(resources.device)
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = resources.model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )

    return _decode_generation(resources, inputs, generated_ids)


def generate_textual_response(
    resources: QwenResources,
    query: str,
    contexts: Sequence[str],
    qa_prompt: str,
    max_new_tokens: int = 512,
) -> str:
    """텍스트 청크를 근거로 한 답변 생성을 수행한다."""
    contexts_str = "\n- ".join(contexts) if contexts else ""
    prompt = (
        "You are tasked with answering a question based on the relevant chunks of a PDF document. "
        "Provide your response in the following format:\n"
        "## Evidence:\n\n"
        "## Chain of Thought:\n\n"
        "## Answer:\n\n"
        "___\n"
        "Instructions:\n\n"
        "1. Evidence Curation: Extract relevant elements (such as paragraphs, tables, figures, charts) from the provided chunks and populate them in the \"Evidence\" section. "
        "For each element, include the type, content, and a brief explanation of its relevance.\n\n"
        "2. Chain of Thought: In the \"Chain of Thought\" section, list out each logical step you take to derive the answer, referencing the evidence where applicable. "
        "You should perform computations if needed.\n\n"
        f"3. Answer: {qa_prompt}\n___\n"
        f"Question: {query}\n___\nContext: {contexts_str}"
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = resources.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = resources.processor(
        text=[text], padding=True, return_tensors="pt"
    )
    device = torch.device(resources.device)
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = resources.model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )

    return _decode_generation(resources, inputs, generated_ids)


def generate_text_only_response(
    resources: QwenResources,
    prompt: str,
    max_new_tokens: int = 1000,
) -> str:
    """텍스트만으로 이뤄진 프롬프트를 실행해 응답 결합 단계에 활용한다."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = resources.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = resources.processor(
        text=[text], padding=True, return_tensors="pt"
    )
    device = torch.device(resources.device)
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = resources.model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )

    return _decode_generation(resources, inputs, generated_ids)


__all__ = [
    "QwenResources",
    "init_qwen",
    "generate_visual_response",
    "generate_textual_response",
    "generate_text_only_response",
]
