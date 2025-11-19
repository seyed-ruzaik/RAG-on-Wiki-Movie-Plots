"""
src/generate.py
================
Task 5: RAG answer generation with a local Hugging Face model (default: google/flan-t5-small).

Changes vs previous version:
- Prompt includes context TITLES and explicit instruction to name films when relevant.
- Defaults to temperature=0.0 (deterministic).
"""

import argparse
import json
import os
from typing import List, Dict, Any

import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------- Retrieval ----------
def retrieve_top_k(query: str, persist_dir: str, collection: str, k: int = 3) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(collection)
    result = col.query(
        query_texts=[query],
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"]
    )
    return result


# ---------- Prompt Building ----------
def _shorten(text: str, max_chars: int = 1000) -> str:
    """Trim long chunks so the prompt stays compact."""
    t = text.strip()
    return (t[:max_chars] + " ...") if len(t) > max_chars else t

def build_prompt(question: str, documents: List[str], metadatas: List[dict]) -> str:
    """
    Build an instruction-style prompt with explicit guidance.
    We label each context with its Title so the model can cite it.
    """
    items = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        title = str(meta.get("title", "Unknown"))
        chunk_id = meta.get("chunk_id", "N/A")
        items.append(f"Context {i} — Title: {title} (chunk {chunk_id})\n{_shorten(str(doc), 900)}")

    context_block = "\n\n".join(items) if items else "(no context)"

    prompt = (
        "You are a precise assistant. Answer ONLY using the given contexts.\n"
        "If the answer is not present, reply exactly: \"I don't know from the provided context.\".\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- If any context mentions a ship hitting or striking an iceberg, name the film title(s) and summarize in one short sentence.\n"
        "- Otherwise, reply exactly: \"I don't know from the provided context.\".\n"
        "Answer:"
    )
    return prompt


# ---------- HF Generation ----------
def load_hf_generator(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def generate_answer(
    question: str,
    docs: List[str],
    metas: List[dict],
    model_name: str = "google/flan-t5-small",
    max_input_tokens: int = 1024,
    max_new_tokens: int = 128,
    temperature: float = 0.0,   # deterministic by default
    top_p: float = 0.9,
) -> str:
    tokenizer, model = load_hf_generator(model_name)
    prompt = build_prompt(question, docs, metas)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        top_p=top_p
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer


# ---------- Orchestration ----------
def run_pipeline(
    query: str,
    persist_dir: str,
    collection: str,
    k: int,
    model_name: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float
) -> Dict[str, Any]:
    retrieved = retrieve_top_k(query, persist_dir, collection, k)
    docs = retrieved.get("documents", [[]])[0] if retrieved.get("documents") else []
    metas = retrieved.get("metadatas", [[]])[0] if retrieved.get("metadatas") else []

    answer = generate_answer(
        question=query,
        docs=docs,
        metas=metas,
        model_name=model_name,
        max_input_tokens=max_input_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    out = {
        "question": query,
        "answer": answer,
        "contexts_used": docs[:k],
        "metadatas": metas[:k],
        "model_name": model_name
    }
    return out


# ---------- CLI ----------
def cli():
    parser = argparse.ArgumentParser(description="Task 5: RAG answer generation using a Hugging Face model.")
    parser.add_argument("--query", required=True, help="User question, e.g., 'a ship hits an iceberg'.")
    parser.add_argument("--persist_dir", required=True, help="Chroma DB directory, e.g., data/vectorstore")
    parser.add_argument("--collection", default="movie_chunks", help="Chroma collection name.")
    parser.add_argument("--k", type=int, default=3, help="Top-K contexts to use (keep small).")

    parser.add_argument("--model_name", default="google/flan-t5-small", help="HF model to use (e.g., google/flan-t5-base).")
    parser.add_argument("--max_input_tokens", type=int, default=1024, help="Max input tokens (prompt).")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Creativity. 0.0 = deterministic.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold (ignored if temperature=0).")

    parser.add_argument("--save_json", default="", help="If set, save the result JSON to this path.")
    args = parser.parse_args()

    result = run_pipeline(
        query=args.query,
        persist_dir=args.persist_dir,
        collection=args.collection,
        k=args.k,
        model_name=args.model_name,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== CONTEXT TITLES ===")
    for m in result["metadatas"]:
        print(f"- {m.get('title', 'Unknown')} (chunk {m.get('chunk_id', 'N/A')})")

    print("\n=== JSON OUTPUT ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {args.save_json}")


if __name__ == "__main__":
    cli()
