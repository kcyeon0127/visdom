from __future__ import annotations

import csv
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from difflib import SequenceMatcher
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from tqdm import tqdm

from .config import VisDoMRAGConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalManager:
    """문서 캐싱·인덱싱·검색 전 과정을 담당하는 유틸리티 클래스."""

    config: VisDoMRAGConfig
    df: pd.DataFrame
    document_cache: Dict[str, List[str]] = field(default_factory=dict)
    vision_model: Optional[object] = None
    vision_processor: Optional[object] = None
    st_embedding_function: Optional[object] = None
    text_model_name: Optional[str] = None
    device: str = field(init=False)

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config.ensure_directories()
        self._initialize_retrievers()

    def _initialize_retrievers(self) -> None:
        cfg = self.config

        if cfg.vision_retriever not in {"colpali", "colqwen"}:
            raise ValueError(f"Unsupported visual retriever: {cfg.vision_retriever}")

        try:
            from colpali_engine.models import ColPali, ColPaliProcessor, ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise ImportError(
                "Install colpali_engine to use visual retrievers."
            ) from exc

        if cfg.vision_retriever == "colpali":
            logger.info("Loading ColPali model for visual indexing on CUDA")
            self.vision_model = ColPali.from_pretrained(
                "vidore/colpali-v1.2",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self.vision_processor = ColPaliProcessor.from_pretrained(
                "vidore/colpali-v1.2"
            )
        else:
            logger.info("Loading ColQwen model for visual indexing on CUDA")
            self.vision_model = ColQwen2.from_pretrained(
                "vidore/colqwen2-v0.1",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self.vision_processor = ColQwen2Processor.from_pretrained(
                "vidore/colqwen2-v0.1"
            )

        if cfg.text_retriever == "bm25":
            logger.info("Using BM25 for text retrieval")
        elif cfg.text_retriever in {"minilm", "mpnet", "bge"}:
            try:
                import chromadb.utils.embedding_functions as embedding_functions
            except ImportError as exc:  # pragma: no cover - optional
                raise ImportError(
                    "Install chromadb and sentence-transformers for dense retrieval."
                ) from exc
            model_map = {
                "minilm": "sentence-transformers/all-MiniLM-L6-v2",
                "mpnet": "sentence-transformers/all-mpnet-base-v2",
                "bge": "BAAI/bge-base-en-v1.5",
            }
            self.text_model_name = model_map[cfg.text_retriever]
            self.st_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.text_model_name,
                device=self.device,
            )
            logger.info("Using %s embeddings for text retrieval", self.text_model_name)
        else:
            raise ValueError(f"Unsupported text retriever: {cfg.text_retriever}")

    # ------------------------------------------------------------------
    # 문서 준비 단계
    # ------------------------------------------------------------------
    def _collect_unique_docs(self) -> List[str]:
        unique_docs = set()
        for _, row in self.df.iterrows():
            if 'documents' in row and isinstance(row['documents'], str):
                try:
                    docs = eval(row['documents'])  # noqa: S307 - dataset field
                    unique_docs.update(docs)
                except Exception:
                    traceback.print_exc()
            if 'doc_path' in row:
                doc_path = row['doc_path']
                if isinstance(doc_path, str) and doc_path.strip():
                    unique_docs.add(Path(doc_path).stem)
        return list(unique_docs)

    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        pages: List[str] = []
        try:
            reader = PdfReader(str(pdf_path))
            pages = [page.extract_text() or "" for page in reader.pages]

            if any(not page.strip() for page in pages):
                logger.info("Falling back to OCR for %s", pdf_path)
                pdf_images = convert_from_path(str(pdf_path))
                pages = []
                for page_num, page_img in enumerate(pdf_images, start=1):
                    text = pytesseract.image_to_string(page_img)
                    pages.append(f"--- Page {page_num} ---\n{text}\n")
        except Exception:
            logger.exception("Error extracting text from %s", pdf_path)
        return pages

    def split_text(self, text: str) -> List[str]:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        return splitter.split_text(text)

    def cache_documents(self) -> Dict[str, List[str]]:
        if self.document_cache:
            return self.document_cache

        logger.info("Caching PDF contents")
        pdf_dir = self.config.data_dir / "docs"
        unique_docs = self._collect_unique_docs()

        desc = f"Caching documents ({self.config.data_dir.name})"
        for doc_id in tqdm(unique_docs, desc=desc):
            candidates = [
                pdf_dir / doc_id,
                pdf_dir / f"{doc_id}.pdf",
                pdf_dir / f"{doc_id.ljust(10, '0')}.pdf",
                pdf_dir / f"{doc_id.split('_')[0]}.pdf",
            ]
            for pdf_path in candidates:
                if pdf_path.exists():
                    self.document_cache[doc_id] = self.extract_text_from_pdf(pdf_path)
                    break
            else:
                logger.warning("No PDF found for document %s", doc_id)
        logger.info("Cached %d documents", len(self.document_cache))
        return self.document_cache

    def identify_document_and_page(
        self, chunk: str
    ) -> Tuple[Optional[str], Optional[int]]:
        max_ratio = 0.0
        best_match: Tuple[Optional[str], Optional[int]] = (None, None)
        for doc_id, pages in self.document_cache.items():
            for page_num, page_text in enumerate(pages):
                ratio = SequenceMatcher(None, chunk, page_text).ratio()
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_match = (doc_id, page_num)
        return best_match

    # ------------------------------------------------------------------
    # 인덱스 구축 단계
    # ------------------------------------------------------------------
    def build_visual_index(self) -> bool:
        cfg = self.config
        logger.info("Building visual index using %s", cfg.vision_retriever)

        pdf_dir = cfg.data_dir / "docs"
        output_dir = cfg.data_dir / "visual_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_docs = self._collect_unique_docs()
        pdf_files = [doc if doc.endswith(".pdf") else f"{doc}.pdf" for doc in unique_docs]

        page_embeddings: Dict[str, torch.Tensor] = {}

        for pdf_file in tqdm(
            pdf_files,
            desc=f"Visual index PDFs ({self.config.data_dir.name})",
        ):
            doc_id = Path(pdf_file).stem
            pdf_path = pdf_dir / pdf_file
            if not pdf_path.exists():
                logger.warning("PDF not found: %s", pdf_path)
                continue
            try:
                pages = convert_from_path(str(pdf_path))
            except Exception:
                logger.exception("Error converting %s to images", pdf_path)
                continue

            for page_idx, page_img in enumerate(pages):
                page_id = f"{doc_id}_{page_idx}"
                try:
                    processed_image = self.vision_processor.process_images([page_img])
                    processed_image = {
                        k: v.to(self.vision_model.device)
                        for k, v in processed_image.items()
                    }
                    with torch.no_grad():
                        embedding = self.vision_model(**processed_image)
                    torch.save(embedding.cpu(), output_dir / f"{page_id}.pt")
                    page_embeddings[page_id] = embedding.cpu()
                except Exception:
                    logger.exception(
                        "Error processing page %d of %s", page_idx, pdf_file
                    )

        query_embeddings: Dict[str, torch.Tensor] = {}
        for _, row in tqdm(
            self.df.iterrows(),
            desc=f"Visual index queries ({self.config.data_dir.name})",
        ):
            q_id = row['q_id']
            question = row['question']
            try:
                processed_query = self.vision_processor.process_queries([question])
                processed_query = {
                    k: v.to(self.vision_model.device)
                    for k, v in processed_query.items()
                }
                with torch.no_grad():
                    embedding = self.vision_model(**processed_query)
                query_embeddings[q_id] = embedding.cpu()
                torch.save(embedding.cpu(), output_dir / f"query_{q_id}.pt")
            except Exception:
                logger.exception("Error embedding query %s", q_id)

        results: List[Dict[str, str | float]] = []
        for q_id, query_emb in tqdm(
            query_embeddings.items(),
            desc=f"Ranking documents ({self.config.data_dir.name})",
        ):
            relevant_docs = self._extract_relevant_docs(q_id)
            if not relevant_docs:
                relevant_docs = [Path(f).stem for f in pdf_files]

            relevant_page_embeddings = {
                page_id: emb
                for page_id, emb in page_embeddings.items()
                if page_id.rsplit('_', 1)[0] in relevant_docs
            }

            if not relevant_page_embeddings:
                logger.warning(
                    "No visual embeddings found for query %s", q_id
                )
                continue

            qs = query_emb
            ds = torch.cat(list(relevant_page_embeddings.values()), dim=0)
            scores = self.vision_processor.score_multi_vector(qs, ds)
            scores = scores.flatten().numpy()

            ranked_docs = np.array(list(relevant_page_embeddings.keys()))
            top_indices = np.argsort(-scores)

            for doc_id, score in zip(ranked_docs[top_indices], scores[top_indices]):
                results.append(
                    {
                        'q_id': q_id,
                        'document_id': doc_id,
                        'score': float(score),
                        'question': self._question_from_id(q_id),
                    }
                )

        with open(cfg.vision_retrieval_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=['q_id', 'document_id', 'score', 'question']
            )
            writer.writeheader()
            writer.writerows(results)

        logger.info("Visual index saved to %s", cfg.vision_retrieval_file)
        return True

    def _question_from_id(self, q_id: str) -> str:
        row = self.df[self.df['q_id'] == q_id]
        if row.empty:
            return ""
        return row.iloc[0]['question']

    def _extract_relevant_docs(self, q_id: str) -> List[str]:
        row = self.df[self.df['q_id'] == q_id]
        if row.empty:
            return []
        relevant_docs: List[str] = []
        entry = row.iloc[0]
        if 'documents' in entry and isinstance(entry['documents'], str):
            try:
                docs = eval(entry['documents'])  # noqa: S307
                relevant_docs = [Path(doc).stem for doc in docs]
            except Exception:
                traceback.print_exc()
                pdf_dir = self.config.data_dir / "docs"
                relevant_docs = [p.stem for p in pdf_dir.glob("*.pdf")]
        elif 'doc_path' in entry:
            doc_path = entry['doc_path']
            if isinstance(doc_path, str) and doc_path.strip():
                relevant_docs.append(Path(doc_path).stem)
        return relevant_docs

    def build_text_index(self) -> bool:
        cfg = self.config
        logger.info("Building text index using %s", cfg.text_retriever)

        if not self.document_cache:
            self.cache_documents()

        all_chunks: List[str] = []
        chunk_to_doc: List[Dict[str, str | int]] = []

        for doc_id, pages in tqdm(
            self.document_cache.items(),
            desc=f"Text index docs ({self.config.data_dir.name})",
        ):
            text = "\n".join(pages)
            chunks = self.split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                arxiv_id, page_num = self.identify_document_and_page(chunk)
                chunk_to_doc.append(
                    {
                        'chunk': chunk,
                        'chunk_pdf_name': arxiv_id or doc_id,
                        'pdf_page_number': page_num or 0,
                    }
                )

        results: List[Dict[str, str | int | float]] = []
        if cfg.text_retriever == "bm25":
            try:
                from rank_bm25 import BM25Okapi
            except ImportError as exc:
                raise ImportError("Install rank_bm25 for BM25 retrieval.") from exc
            bm25_model = BM25Okapi([chunk.split() for chunk in all_chunks])
            for _, row in tqdm(self.df.iterrows(), desc="Processing BM25 queries"):
                q_id = row['q_id']
                question = row['question']
                scores = bm25_model.get_scores(question.split())
                top_indices = np.argsort(-scores)[: cfg.top_k * 2]
                for rank, idx in enumerate(top_indices, start=1):
                    mapping = chunk_to_doc[idx]
                    results.append(
                        {
                            'q_id': q_id,
                            'question': question,
                            'chunk': all_chunks[idx],
                            'chunk_pdf_name': mapping['chunk_pdf_name'],
                            'pdf_page_number': mapping['pdf_page_number'],
                            'rank': rank,
                            'score': float(scores[idx]),
                        }
                    )
        else:
            if self.st_embedding_function is None:
                raise ImportError(
                    "Sentence-transformer embedding function is not initialized."
                )
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError("Install chromadb for dense retrieval.") from exc
            chroma_client = chromadb.Client()
            collection = chroma_client.create_collection(
                f"st_col_{uuid.uuid4().hex[:8]}",
                embedding_function=self.st_embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            collection.add(
                documents=all_chunks,
                ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            )
            for _, row in tqdm(
                self.df.iterrows(), desc=f"Processing {cfg.text_retriever} queries"
            ):
                q_id = row['q_id']
                question = row['question']
                query_results = collection.query(
                    query_texts=[question], n_results=cfg.top_k * 2
                )
                ids = [int(i.split('_')[1]) for i in query_results['ids'][0]]
                distances = query_results['distances'][0]
                for rank, (chunk_idx, distance) in enumerate(
                    zip(ids, distances), start=1
                ):
                    mapping = chunk_to_doc[chunk_idx]
                    results.append(
                        {
                            'q_id': q_id,
                            'question': question,
                            'chunk': all_chunks[chunk_idx],
                            'chunk_pdf_name': mapping['chunk_pdf_name'],
                            'pdf_page_number': mapping['pdf_page_number'],
                            'rank': rank,
                            'score': float(1.0 - distance),
                        }
                    )

        with open(cfg.text_retrieval_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    'q_id',
                    'question',
                    'chunk',
                    'chunk_pdf_name',
                    'pdf_page_number',
                    'rank',
                    'score',
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        logger.info("Text index saved to %s", cfg.text_retrieval_file)
        return True

    # ------------------------------------------------------------------
    # 검색 수행 단계
    # ------------------------------------------------------------------
    def retrieve_visual_contexts(self, query_id: str) -> List[Dict[str, object]]:
        cfg = self.config
        if cfg.force_reindex or not cfg.vision_retrieval_file.exists():
            if not self.build_visual_index():
                return []

        df_retrieval = pd.read_csv(cfg.vision_retrieval_file)
        query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
        if query_rows.empty:
            logger.warning("No visual contexts found for %s", query_id)
            return []

        top_k_rows = query_rows.nlargest(cfg.top_k, 'score')
        pdf_dir = cfg.data_dir / "docs"
        pages: List[Dict[str, object]] = []
        for _, row in top_k_rows.iterrows():
            document_id = row['document_id']
            base_doc_id, page_number = document_id.rsplit('_', 1)
            page_idx = int(page_number)
            pdf_path = pdf_dir / f"{base_doc_id}.pdf"
            if not pdf_path.exists():
                logger.warning("PDF file not found: %s", pdf_path)
                continue
            try:
                pdf_images = convert_from_path(str(pdf_path))
            except Exception:
                logger.exception("Unable to render %s", pdf_path)
                continue
            if page_idx >= len(pdf_images):
                logger.warning("Page %d out of range for %s", page_idx, pdf_path)
                continue
            pages.append(
                {
                    'image': pdf_images[page_idx],
                    'document_id': document_id,
                    'page_number': page_idx,
                }
            )
        return pages

    def retrieve_textual_contexts(self, query_id: str) -> List[Dict[str, object]]:
        cfg = self.config
        if cfg.force_reindex or not cfg.text_retrieval_file.exists():
            if not self.build_text_index():
                return []

        df_retrieval = pd.read_csv(cfg.text_retrieval_file)
        query_rows = df_retrieval[df_retrieval['q_id'] == query_id]
        if query_rows.empty:
            logger.warning("No textual contexts found for %s", query_id)
            return []

        top_k_rows = (
            query_rows.sort_values(by='rank', ascending=True).head(cfg.top_k)
        )
        contexts = []
        for _, row in top_k_rows.iterrows():
            contexts.append(
                {
                    'chunk': row['chunk'],
                    'chunk_pdf_name': row.get('chunk_pdf_name', 'unknown'),
                    'pdf_page_number': int(row.get('pdf_page_number', 0)),
                }
            )
        return contexts


__all__ = ["RetrievalManager"]
