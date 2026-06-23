"""
MediBot Document Ingestion Pipeline — Standalone script.

Workflow:
  1. Scan mediassist_data/ for PDFs and Markdown documents.
  2. Group files into domain collections and map role access lists.
  3. Parse documents using Docling (disabling OCR) and generate layout-aware chunks.
  4. Generate sparse BM25 embeddings via FastEmbed.
  5. Encode chunks using a SentenceTransformer dense embedding model.
  6. Index both dense and sparse vectors in a local Qdrant collection.

Usage:
    python ingest.py
"""
import os
import sys
from typing import List, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from qdrant_client.models import PointStruct

# Add backend to path for package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATA_DIR, COLLECTION_NAME, DENSE_EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL, ACCESS_MATRIX
)
from adapters.embedding_adapter import EmbeddingAdapter
from adapters.qdrant_adapter import QdrantAdapter
from utils.logger import get_logger

logger = get_logger("ingest")


def get_chunk_type(doc_chunk) -> str:
    """
    Detects chunk type from doc_items types.
    Checks all items in the chunk to flag tables, code blocks, or headings.
    """
    try:
        if not doc_chunk.meta.doc_items or len(doc_chunk.meta.doc_items) == 0:
            return "text"

        types = [type(item).__name__.lower() for item in doc_chunk.meta.doc_items]

        if any("table" in t for t in types):
            return "table"
        elif any("code" in t for t in types):
            return "code"
        elif any("heading" in t or "section" in t or "header" in t for t in types):
            return "heading"
    except Exception:
        pass
    return "text"


def main():
    """Main document ingestion and indexing pipeline."""
    logger.info("Initializing document ingestion pipeline...")

    # 1. Scanning directories for documents
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        if folder_name not in ACCESS_MATRIX:
            continue

        for file in files:
            if file.endswith(".pdf") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                documents.append({
                    "path": file_path,
                    "filename": file,
                    "collection": folder_name,
                    "access_roles": ACCESS_MATRIX[folder_name]["access_roles"]
                })

    logger.info(f"Found {len(documents)} documents to ingest.")

    # Configure Docling (disable OCR to avoid environment issues, enable PDF/MD)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.MD],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    chunker = HierarchicalChunker()

    # 2. Convert and chunk documents
    all_chunks_raw = []
    for doc_info in documents:
        logger.info(f"Parsing: {doc_info['filename']} in collection: {doc_info['collection']}")
        try:
            result = converter.convert(doc_info["path"])
            doc_obj = result.document

            doc_chunks = list(chunker.chunk(doc_obj))
            logger.info(f"Generated {len(doc_chunks)} chunks for {doc_info['filename']}")

            for c in doc_chunks:
                headings = c.meta.headings or []
                content = c.text.strip()
                breadcrumb = " > ".join(headings)
                chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

                section_title = headings[-1] if headings else "General"

                all_chunks_raw.append({
                    "content": content,
                    "chunk_text": chunk_text,
                    "source_document": doc_info["filename"],
                    "collection": doc_info["collection"],
                    "access_roles": doc_info["access_roles"],
                    "section_title": section_title,
                    "chunk_type": get_chunk_type(c)
                })
        except Exception as e:
            logger.error(f"Error parsing {doc_info['filename']}: {e}")

    logger.info(f"Total processed chunks across all documents: {len(all_chunks_raw)}")

    if not all_chunks_raw:
        logger.warning("No chunks to index.")
        return

    # 3. Initialize embedding adapter and generate embeddings
    embedding_adapter = EmbeddingAdapter()
    chunk_texts = [c["chunk_text"] for c in all_chunks_raw]

    logger.info("Generating sparse embeddings...")
    sparse_embeddings_raw = embedding_adapter.encode_sparse_batch(chunk_texts)

    logger.info("Generating dense embeddings...")
    dense_vectors = embedding_adapter.encode_dense_batch(chunk_texts, show_progress=True)

    # 4. Initialize Qdrant and create collection
    qdrant_adapter = QdrantAdapter()
    qdrant_adapter.create_collection()

    # 5. Build and upload points
    logger.info("Building indexed points...")
    points = []
    for i, chunk in enumerate(all_chunks_raw):
        dense_vec = dense_vectors[i].tolist()

        sparse_emb = sparse_embeddings_raw[i]
        sparse_vec = {
            "indices": sparse_emb.indices.tolist(),
            "values": sparse_emb.values.tolist()
        }

        points.append(
            PointStruct(
                id=i,
                vector={
                    "text-dense": dense_vec,
                    "text-sparse": sparse_vec
                },
                payload={
                    "content": chunk["content"],
                    "chunk_text": chunk["chunk_text"],
                    "source_document": chunk["source_document"],
                    "collection": chunk["collection"],
                    "access_roles": chunk["access_roles"],
                    "section_title": chunk["section_title"],
                    "chunk_type": chunk["chunk_type"]
                }
            )
        )

    qdrant_adapter.upload_points(points)

    # 6. Print stats for validation
    logger.info("==================================================")
    logger.info("Ingestion Validation Summary:")
    logger.info(f"Total chunks indexed: {len(all_chunks_raw)}")

    collections_counts = {}
    for chunk in all_chunks_raw:
        c = chunk["collection"]
        collections_counts[c] = collections_counts.get(c, 0) + 1
    logger.info("Chunks per collection:")
    for c, count in collections_counts.items():
        logger.info(f"  - {c}: {count}")

    type_counts = {}
    for chunk in all_chunks_raw:
        t = chunk["chunk_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("Chunks per chunk type:")
    for t, count in type_counts.items():
        logger.info(f"  - {t}: {count}")
    logger.info("==================================================")
    logger.info("Document ingestion and indexing completed successfully!")


if __name__ == "__main__":
    main()
