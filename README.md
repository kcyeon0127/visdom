# VisDoM: 멀티모달 문서 QA 파이프라인

NAACL 2025 논문 *VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation* 에서 사용한 데이터 분할과 `VisDoMRAG` 파이프라인을 제공합니다. 테이블·차트·슬라이드 등 시각 요소가 풍부한 문서를 대상으로 멀티모달 RAG 실험을 재현할 수 있습니다.

## 저장소 구성
- `feta_tab`, `paper_tab`, `scigraphvqa`, `slidevqa`, `spiqa`: VisDoMBench 데이터 분할과 PDF
- `visdomrag/`: 모듈화된 파이프라인 코드 (Qwen 전용)
- `visdomrag_qwen.ipynb`: 단일 데이터셋 실험 노트북 (GPU 전환 셀 포함)
- `run_all_datasets.py`: 여러 데이터셋을 순차 실행하고 F1을 기록하는 CLI 스크립트
- `qwen_eval.ipynb`: 생성된 JSON을 불러 F1을 확인하는 평가 노트북

## 핵심 기능
1. **시각 검색(ColPali / ColQwen)**: PDF → 이미지 변환 → 페이지 임베딩 생성 및 저장(`visual_embeddings/*.pt`)
2. **텍스트 검색(BM25, MiniLM, MPNet, BGE)**: PyPDF2+OCR로 추출한 텍스트를 청킹한 뒤 CSV 인덱스로 저장(`retrieval/retrieval_*.csv`)
3. **문서 텍스트 캐시**: `retrieval/document_cache.pkl`에 PDF 텍스트를 저장/불러와 재실행 시 시간을 절약 ( `force_reindex=True` 이면 재구축 )
4. **Qwen 7B 추론**: 이미지/텍스트 응답을 각각 생성하고 결합, JSON으로 저장 (`outputs/<dataset>/qwen_*`)
5. **GPU 분리 실행 지원**: 노트북의 `set_gpu()` 셀 또는 `run_all_datasets.py --index-gpu --pipeline-gpu` 옵션으로 ColPali와 Qwen을 다른 GPU에 배치

## 실행 방법
### 노트북에서 단일 쿼리/데이터셋 실험
1. `visdomrag_qwen.ipynb`를 열고 맨 위 `set_gpu()` 셀에서 ColPali용 GPU를 설정합니다.
2. 인덱싱이 끝나면 Qwen 섹션 앞 셀을 실행하여 캐시 정리 후 Qwen용 GPU(예: `set_gpu([2,3])`)를 지정합니다.
3. `sample_id`를 선택해 `process_query` 셀을 실행하면 해당 q_id의 비주얼/텍스트/결합 응답 JSON이 생성됩니다.

### 여러 데이터셋 일괄 처리
```bash
# 인덱싱만 GPU1에서 수행
python run_all_datasets.py --phase index --index-gpu 1
# Qwen 파이프라인을 GPU2,3에서 수행
python run_all_datasets.py --phase pipeline --pipeline-gpu 2,3
```
- `--phase all`을 사용하면 한 번에 두 단계를 순차 실행합니다.
- 진행 상황은 `Indexing datasets ...`, `Running pipelines ...` tqdm으로 표시되며, 결과 F1은 `run_all_summary.json`에 저장됩니다.

### 전처리만 선행 실행
Qwen 추론 없이 문서 캐시/시각·텍스트 인덱스만 미리 만들고 싶다면 `run_preprocessing.py`를 사용합니다.
```bash
python run_preprocessing.py --dataset feta_tab --gpu 1
# 캐시/인덱스를 다시 만들고 싶다면 --force 추가
```
실행이 끝나면 `retrieval/document_cache.pkl`, `retrieval/retrieval_*.csv`, `visual_embeddings/*.pt`가 생성되어 나중에 Qwen 추론만 빠르게 수행할 수 있습니다.

### 결과 평가
노트북 대신 CLI로 평가하려면:
```bash
python eval.py outputs/feta_tab_qwen/qwen_visdmrag
```
혹은 `qwen_eval.ipynb`에서 `EVAL_DIR`를 지정하고 셀을 실행하면 파일별/평균 F1을 표로 확인할 수 있습니다.

## 사용자 정의 옵션 (예시)
```python
from visdomrag import VisDoMRAGConfig

config = VisDoMRAGConfig(
    data_dir=Path("./feta_tab"),
    output_dir=Path("./outputs/feta_tab_qwen"),
    vision_retriever="colpali",
    text_retriever="bm25",
    top_k=5,
    chunk_size=3000,
    chunk_overlap=300,
    force_reindex=False,
)
```
- `force_reindex=True`로 설정하면 시각/텍스트 인덱스 및 `document_cache.pkl`을 다시 생성합니다.

## 의존성 요약
- PyTorch, pandas, numpy, tqdm
- PDF/OCR: pdf2image, PyPDF2, pytesseract (→ OS 패키지로 poppler, tesseract 설치 필요)
- 텍스트 검색: langchain 또는 langchain-text-splitters, chromadb, rank-bm25
- 시각 검색: colpali_engine (ColPali/ColQwen)
- Qwen: transformers, qwen-vl-utils (FlashAttention2 사용 시 `flash-attn --no-build-isolation` 설치 권장)

## 데이터셋 통계
| 데이터셋 | 도메인 | 주요 콘텐츠 | 질문 수 | 문서 수 |
|----------|--------|-------------|---------|---------|
| PaperTab | Wikipedia | 테이블+텍스트 | 377 | 297 |
| FetaTab | 논문 | 테이블 | 350 | 300 |
| SciGraphQA | 논문 | 차트 | 407 | 319 |
| SPIQA | 논문 | 테이블+차트 | 586 | 117 |
| SlideVQA | 슬라이드 | 슬라이드 | 551 | 244 |
| VisDoMBench | 통합 | 테이블/차트/슬라이드 | 2271 | 1277 |

## 인용
```bibtex
@misc{suri2024visdommultidocumentqavisually,
      title={VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation},
      author={Manan Suri and Puneet Mathur and Franck Dernoncourt and Kanika Goswami and Ryan A. Rossi and Dinesh Manocha},
      year={2024},
      eprint={2412.10704},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
