import os
import re
import json
import math
from typing import List, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker import HierarchicalChunker
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams, PointStruct

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "mediassist_data")
QDRANT_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "qdrant_db")
COLLECTION_NAME = "medibot"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BM25_MODEL_PATH = os.path.join(SCRIPT_DIR, "mediassist_data", "bm25_model.json")


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

class BM25Vectorizer:
    """
    Custom Implementation of Best Match 25 (BM25) term weighting for sparse vector retrieval.
    
    This vectorizer calculates term frequencies, document frequencies, and inverse 
    document frequencies (IDF) locally over the parsed chunk corpus. It represents 
    each chunk as a sparse vector of keyword weights, which is compatible with Qdrant.
    
    Using BM25 alongside dense vector search allows the system to support hybrid search, 
    combining semantic understanding with exact keyword matches for specific drug codes, 
    medical terms, or equipment model numbers.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.vocab = {}  # word -> index
        self.idf = {}    # word_index -> idf
        self.avg_doc_len = 0.0
        self.doc_count = 0

    def _tokenize(self, text: str) -> List[str]:
        # Simple word tokenization, removing punctuation and lowercasing
        return re.findall(r"\w+", text.lower())

    def fit(self, corpus: List[str]):
        self.doc_count = len(corpus)
        if self.doc_count == 0:
            return

        doc_lengths = []
        doc_term_freqs = []  # list of dicts: word_index -> freq
        word_doc_counts = {} # word -> count of docs containing it

        for doc in corpus:
            tokens = self._tokenize(doc)
            doc_lengths.append(len(tokens))
            
            tf = {}
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
                w_idx = self.vocab[token]
                tf[w_idx] = tf.get(w_idx, 0) + 1
            
            doc_term_freqs.append(tf)
            for w_idx in tf.keys():
                word_doc_counts[w_idx] = word_doc_counts.get(w_idx, 0) + 1

        self.avg_doc_len = sum(doc_lengths) / self.doc_count

        # Compute IDF for each word index
        for w_idx, count in word_doc_counts.items():
            # BM25 IDF formulation
            self.idf[w_idx] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)

    def transform(self, text: str, doc_len: int = None) -> Dict[str, Any]:
        """
        Converts text into Qdrant SparseVector compatible dict format:
        { "indices": [int, ...], "values": [float, ...] }
        """
        tokens = self._tokenize(text)
        if not doc_len:
            doc_len = len(tokens)

        tf = {}
        for token in tokens:
            if token in self.vocab:
                w_idx = self.vocab[token]
                tf[w_idx] = tf.get(w_idx, 0) + 1

        indices = []
        values = []
        for w_idx, freq in tf.items():
            idf_val = self.idf.get(w_idx, 0.0)
            # BM25 term weight
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
            weight = idf_val * (numerator / denominator)
            
            if weight > 0.0:
                indices.append(w_idx)
                values.append(weight)

        # Qdrant client expects sorted indices for efficiency
        sorted_pairs = sorted(zip(indices, values))
        if sorted_pairs:
            indices, values = zip(*sorted_pairs)
            return {"indices": list(indices), "values": list(values)}
        return {"indices": [], "values": []}

    def save(self, filepath: str):
        data = {
            "k1": self.k1,
            "b": self.b,
            "vocab": self.vocab,
            "idf": {str(k): v for k, v in self.idf.items()},
            "avg_doc_len": self.avg_doc_len,
            "doc_count": self.doc_count
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vectorizer = cls(k1=data["k1"], b=data["b"])
        vectorizer.vocab = data["vocab"]
        vectorizer.idf = {int(k): v for k, v in data["idf"].items()}
        vectorizer.avg_doc_len = data["avg_doc_len"]
        vectorizer.doc_count = data["doc_count"]
        return vectorizer

def get_chunk_type(doc_chunk) -> str:
    """Detects chunk type from doc_items types."""
    try:
        if not doc_chunk.meta.doc_items or len(doc_chunk.meta.doc_items) == 0:
            return "text"
        first_item = doc_chunk.meta.doc_items[0]
        type_name = type(first_item).__name__.lower()
        
        if "table" in type_name:
            return "table"
        elif "heading" in type_name or "section" in type_name or "header" in type_name:
            return "heading"
        elif "code" in type_name:
            return "code"
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
    
    # Configure Docling (disable OCR to avoid environment issues)
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    converter = DocumentConverter(
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

    # 3. Fit BM25 Model on the raw chunk text corpus
    print("Fitting BM25 vectorizer...")
    bm25 = BM25Vectorizer()
    bm25.fit([c["chunk_text"] for c in all_chunks_raw])
    bm25.save(BM25_MODEL_PATH)
    print(f"BM25 vocabulary size: {len(bm25.vocab)}. Saved model to {BM25_MODEL_PATH}")

    # 4. Dense Embeddings Generation
    print(f"Generating dense embeddings using model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chunk_texts = [c["chunk_text"] for c in all_chunks_raw]
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

    # 6. Upload Points to Qdrant
    print("Uploading indexed points to Qdrant...")
    points = []
    for i, chunk in enumerate(all_chunks_raw):
        dense_vec = dense_vectors[i].tolist()
        # Generate BM25 sparse vector
        sparse_vec = bm25.transform(chunk["chunk_text"])
        
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

    print("Document ingestion and indexing completed successfully!")

if __name__ == "__main__":
    main()
