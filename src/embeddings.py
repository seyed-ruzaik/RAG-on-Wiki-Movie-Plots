"""
src/embeddings.py (Chroma version)
=================================
Task 3: Turn chunked text into embeddings and store them in a local Chroma vector DB.

Usage (local SBERT model, no API key needed):
    python src/embeddings.py \
        --input data/processed/chunks.csv \
        --persist_dir data/vectorstore \
        --collection movie_chunks \
        --batch_size 128

What it does:
1) Reads chunked dataset (expects columns: title, chunk_id, chunk_text).
2) Encodes chunk_text using sentence-transformers (all-MiniLM-L6-v2 by default).
3) Upserts embeddings + metadata into a persistent Chroma collection.

Result:
A folder at --persist_dir (e.g., data/vectorstore) that stores the vectors for later retrieval.
"""

import os
import argparse
import uuid
import pandas as pd

import chromadb
from chromadb.config import Settings


# ------------------------------
# Load the encoder (SBERT)
# ------------------------------
def _load_sbert(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a small, fast local sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ------------------------------
# Get or create a Chroma collection
# ------------------------------
def get_or_create_collection(persist_dir: str, collection_name: str):
    """Create or load a persistent Chroma collection (cosine distance)."""
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )


# ------------------------------
# Main function: build embeddings
# ------------------------------
def build_embeddings(input_csv: str, persist_dir: str, collection_name: str, batch_size: int = 128):
    # 1) Load chunked data
    df = pd.read_csv(input_csv)
    required = {"title", "chunk_id", "chunk_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {missing}. Expected: {required}")

    # 2) Load local SBERT encoder
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    encoder = _load_sbert()

    # 3) Get Chroma collection
    col = get_or_create_collection(persist_dir, collection_name)

    # 4) Prepare lists for Chroma
    ids, docs, metas = [], [], []
    for _, row in df.iterrows():
        ids.append(f"{row['title']}__{row['chunk_id']}__{uuid.uuid4().hex}")
        docs.append(str(row["chunk_text"]))
        metas.append({
            "title": str(row["title"]),
            "chunk_id": int(row["chunk_id"]),
            "original_plot_word_count": int(row.get("original_plot_word_count", -1))
        })

    # 5) Batch encode + upsert into Chroma
    n = len(docs)
    print(f"Encoding and upserting {n} chunks...")
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metas[start:end]

        # Encode into vectors
        vectors = encoder.encode(batch_docs, show_progress_bar=False).tolist()

        # Upsert to Chroma
        col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=vectors
        )
        print(f"Upserted {end}/{n}")
        start = end

    print(f"✅ Done. Stored in '{persist_dir}' (collection: {collection_name})")


# ------------------------------
# CLI entry point
# ------------------------------
def cli():
    parser = argparse.ArgumentParser(description="Task 3 (Chroma): build embeddings for chunked text.")
    parser.add_argument("--input", required=True, help="Path to chunked CSV (e.g., data/processed/chunks.csv)")
    parser.add_argument("--persist_dir", required=True, help="Folder to store Chroma DB (e.g., data/vectorstore)")
    parser.add_argument("--collection", default="movie_chunks", help="Chroma collection name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for encoding")
    args = parser.parse_args()

    build_embeddings(args.input, args.persist_dir, args.collection, args.batch_size)


if __name__ == "__main__":
    cli()
