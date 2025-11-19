"""
src/retrieve.py
================
Task 4: Retrieval from the Chroma vector database.

Goal
-----
Given a natural-language query (e.g., "ship sinks after hitting iceberg"),
return the top-K most similar chunks from the embedded collection we built in Task 3.

What this script does
---------------------
1) Opens your persistent Chroma DB (created in Task 3).
2) Runs a similarity search using your text query.
3) Prints a human-readable preview AND (optionally) a JSON blob you can pass into Task 5.

Usage (PowerShell):
-------------------
# Multiline (PowerShell uses the backtick ` for line continuation)
python src\retrieve.py `
  --query "a ship hits an iceberg" `
  --persist_dir data\vectorstore `
  --collection movie_chunks `
  --k 3 `
  --return_json

# One-liner
python src\retrieve.py --query "a ship hits an iceberg" --persist_dir data\vectorstore --collection movie_chunks --k 3 --return_json

Notes:
- --return_json is optional; when provided, prints a JSON object with contexts for Task 5.
"""

import argparse
import json
import os
from typing import List, Dict, Any

import chromadb


# ------------------------------
# Helper: pretty print top results
# ------------------------------
def pretty_print_results(results: Dict[str, Any], max_chars_per_doc: int = 300) -> None:
    """
    Print a simple, readable view of the top matches.

    Args:
        results: Dict returned by collection.query (documents, metadatas, distances, ids).
        max_chars_per_doc: Truncate long chunks so the terminal output stays tidy.
    """
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])
    dists = results.get("distances", [])  # cosine distances (lower is better)

    if not docs:
        print("No results found.")
        return

    # The outer list is per-query; we usually pass a single query, so index 0.
    docs = docs[0]
    metas = metas[0] if metas else [{}] * len(docs)
    dists = dists[0] if dists else [None] * len(docs)

    print("\n=== Top Results ===")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        text = str(doc) if doc is not None else ""
        snippet = (text[:max_chars_per_doc] + "...") if len(text) > max_chars_per_doc else text
        title = meta.get("title", "Unknown")
        chunk_id = meta.get("chunk_id", "N/A")
        dist_txt = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"

        print(f"\n[{i}] Title: {title} | Chunk ID: {chunk_id} | Distance: {dist_txt}")
        print(f"{snippet}")


# ------------------------------
# Helper: build JSON for Task 5
# ------------------------------
def build_task5_json(query: str, results: Dict[str, Any], max_contexts: int) -> Dict[str, Any]:
    """
    Create a minimal JSON payload that Task 5 can consume.

    Returns a dict like:
    {
      "question": "...",
      "contexts": ["chunk text 1", "chunk text 2", ...]   # up to max_contexts
    }
    """
    docs = results.get("documents", [[]])
    contexts = docs[0][:max_contexts] if docs and docs[0] else []
    # Always cast to str for safety
    contexts = [str(c) for c in contexts]
    return {
        "question": query,
        "contexts": contexts
    }


# ------------------------------
# Core retrieval function
# ------------------------------
def retrieve(
    query: str,
    persist_dir: str,
    collection_name: str,
    k: int = 3,
    include_distances: bool = True
) -> Dict[str, Any]:
    """
    Connect to Chroma, run a query, return results.

    Args:
        query: Natural-language query string.
        persist_dir: Folder where Chroma DB was persisted in Task 3.
        collection_name: Name of the collection (e.g., 'movie_chunks').
        k: How many results to return.
        include_distances: Whether to return cosine distances.

    Returns:
        Dict in Chroma query() format: {"ids": ..., "documents": ..., "metadatas": ..., "distances": ...}
    """
    if k < 1:
        raise ValueError("--k must be >= 1")

    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Chroma persist directory not found: '{persist_dir}'. "
            f"Did you run Task 3 and set --persist_dir correctly?"
        )

    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load collection '{collection_name}'. "
            f"Make sure Task 3 created it and the name matches."
        ) from e

    include_fields = ["documents", "metadatas"]
    if include_distances:
        include_fields.append("distances")

    # Run similarity search for the single query (wrapped in a list)
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=include_fields
    )
    return results


# ------------------------------
# CLI entry point
# ------------------------------
def cli():
    parser = argparse.ArgumentParser(description="Task 4: Retrieve top-K chunks from Chroma by semantic similarity.")
    parser.add_argument("--query", required=True, help="Your natural-language search (e.g., 'a ship hits an iceberg').")
    parser.add_argument("--persist_dir", required=True, help="Path to Chroma DB directory (e.g., data/vectorstore).")
    parser.add_argument("--collection", default="movie_chunks", help="Chroma collection name.")
    parser.add_argument("--k", type=int, default=3, help="How many top results to return.")
    parser.add_argument("--max_chars", type=int, default=300, help="Max characters per chunk when printing.")
    parser.add_argument("--return_json", action="store_true", help="If set, prints a JSON with contexts for Task 5.")
    parser.add_argument("--max_contexts", type=int, default=3, help="Limit contexts in Task 5 JSON.")
    args = parser.parse_args()

    # 1) Run retrieval
    results = retrieve(
        query=args.query,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        k=args.k,
        include_distances=True
    )

    # 2) Human-readable preview
    pretty_print_results(results, max_chars_per_doc=args.max_chars)

    # 3) Optional JSON for Task 5
    if args.return_json:
        payload = build_task5_json(args.query, results, max_contexts=args.max_contexts)
        print("\n=== JSON (Task 5 input) ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
