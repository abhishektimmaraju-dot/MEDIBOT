import os
import re
from typing import List, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct
from fastembed import SparseTextEmbedding

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "mediassist_data")
QDRANT_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "qdrant_db")
COLLECTION_NAME = "medibot"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Access Matrix definition
ACCESS_MATRIX = {
    "general": {
        "access_roles": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    },
    "clinical": {
        "access_roles": ["doctor", "admin"],
    },
    "nursing": {
        "access_roles": ["nurse", "doctor", "admin"],
    },
    "billing": {
        "access_roles": ["billing_executive", "admin"],
    },
    "equipment": {
        "access_roles": ["technician", "admin"],
    }
}



def get_chunk_type(doc_chunk) -> str:
    """
    Detects chunk type from doc_items types.
    Checks all items in the chunk to flag tables, code blocks, or headings.
    """
    try:
        if not doc_chunk.meta.doc_items or len(doc_chunk.meta.doc_items) == 0:
            return "text"
        
        # Check all items in the chunk for table or other structural components
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
    """
    Main document ingestion and indexing pipeline.
    
    Workflow:
    1. Scan `mediassist_data/` for PDFs and Markdown documents.
    2. Group files into domain collections and map roles access lists.
    3. Parse documents using Docling (disabling OCR) and generate layout-aware chunks.
    4. Fit the custom BM25 vectorizer over the chunk text corpus.
    5. Encode chunks using a SentenceTransformer dense embedding model.
    6. Index both dense and sparse vectors in a local Qdrant collection.
    """
    print("Initializing document ingestion pipeline...")
    
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
                
    print(f"Found {len(documents)} documents to ingest.")
    
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
        print(f"Parsing: {doc_info['filename']} in collection: {doc_info['collection']}...")
        try:
            result = converter.convert(doc_info["path"])
            doc_obj = result.document

            
            doc_chunks = list(chunker.chunk(doc_obj))
            print(f"Generated {len(doc_chunks)} chunks for {doc_info['filename']}.")
            
            for c in doc_chunks:
                headings = c.meta.headings or []
                content = c.text.strip()
                breadcrumb = " > ".join(headings)
                chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content
                
                # Determine parent heading as section title
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
            print(f"Error parsing {doc_info['filename']}: {e}")
            
    print(f"Total processed chunks across all documents: {len(all_chunks_raw)}")
    
    if not all_chunks_raw:
        print("No chunks to index.")
        return

    # 3. Sparse Embeddings Generation using FastEmbed
    print("Generating sparse embeddings using FastEmbed SparseTextEmbedding...")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    chunk_texts = [c["chunk_text"] for c in all_chunks_raw]
    sparse_embeddings_raw = list(sparse_model.embed(chunk_texts))

    # 4. Dense Embeddings Generation
    print(f"Generating dense embeddings using model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    dense_vectors = embedder.encode(chunk_texts, show_progress_bar=True)
    
    # 5. Initialize Qdrant Collection
    print(f"Initializing local Qdrant database at {QDRANT_PATH}...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    
    # Recreate the collection
    if qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        qdrant_client.delete_collection(COLLECTION_NAME)
        
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "text-dense": VectorParams(
                size=384,  # all-MiniLM-L6-v2 dimension
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=True
                )
            )
        }
    )
    print(f"Created Qdrant collection '{COLLECTION_NAME}' configured for hybrid search (dense + sparse).")

    # Create keyword payload index on access_roles for rapid, secure metadata filtering
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="access_roles",
        field_schema="keyword"
    )
    print("Created keyword payload index on 'access_roles'.")

    # 6. Upload Points to Qdrant
    print("Uploading indexed points to Qdrant...")
    points = []
    for i, chunk in enumerate(all_chunks_raw):
        dense_vec = dense_vectors[i].tolist()
        
        # Extract sparse vectors
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
        
    # Batch upload points
    batch_size = 100
    for offset in range(0, len(points), batch_size):
        batch = points[offset:offset + batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=batch
        )
        print(f"Uploaded batch {offset // batch_size + 1}/{(len(points) - 1) // batch_size + 1}...")

    # 7. Print stats for validation
    print("\n==================================================")
    print("Ingestion Validation Summary:")
    print(f"Total chunks indexed: {len(all_chunks_raw)}")
    
    # Print count per collection
    collections_counts = {}
    for chunk in all_chunks_raw:
        c = chunk["collection"]
        collections_counts[c] = collections_counts.get(c, 0) + 1
    print("Chunks per collection:")
    for c, count in collections_counts.items():
        print(f"  - {c}: {count}")
        
    # Print count per chunk type
    type_counts = {}
    for chunk in all_chunks_raw:
        t = chunk["chunk_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("Chunks per chunk type:")
    for t, count in type_counts.items():
        print(f"  - {t}: {count}")
    print("==================================================")
    print("Document ingestion and indexing completed successfully!")

if __name__ == "__main__":
    main()
