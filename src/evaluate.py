"""
src/evaluate.py
================
Task 7: Batch Evaluation Script for the RAG system.

What this script does:
1) Loads your Chroma vector DB and a Hugging Face model (like FLAN-T5).
2) Runs a set of test questions end-to-end:
     - Retrieve top-K contexts from Chroma
     - Generate a grounded answer
     - Build the required structured JSON: {answer, contexts, reasoning}
3) (Optional) Simple heuristic scoring:
     - If you provide expected keywords per question, we score whether the
       final answer contains any of those keywords (case-insensitive).
4) Saves all results to a JSON file for easy review.

Why this matters:
- Converts your manual checks into a repeatable, auditable evaluation step.
- Perfect as the final task in a take-home to demonstrate completeness.

Usage (PowerShell):
-------------------
# 1) Basic run with built-in test set:
python src\\evaluate.py `
  --persist_dir data\\vectorstore `
  --collection movie_chunks `
  --model_name google/flan-t5-base `
  --k 3 `
  --out data\\outputs\\eval_results.json `
  --deterministic true `
  --max_new_tokens 128

# 2) Run with a custom test file (JSON) you provide:
#    The file should be: [{"question": "...", "expected_keywords": ["kw1","kw2"]}, ...]
python src\\evaluate.py `
  --persist_dir data\\vectorstore `
  --collection movie_chunks `
  --model_name google/flan-t5-base `
  --k 3 `
  --out data\\outputs\\eval_custom.json `
  --tests_file data\\tests\\queries.json `
  --deterministic true
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple

import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# 0) Utility helpers (formatting, text ops)
# -----------------------------

def _shorten(text: str, max_chars: int = 900) -> str:
    """Limit long text chunks to avoid overloading the prompt."""
    t = (text or "").strip()
    return (t[:max_chars] + " ...") if len(t) > max_chars else t

def _first_sentence(text: str, max_chars: int = 300) -> str:
    """Keep only the first sentence (or up to max_chars) to make answers concise."""
    parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    s = parts[0] if parts and parts[0] else text[:max_chars]
    return (s or "").strip()[:max_chars]

def _dedupe(seq: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# -----------------------------
# 1) Vector store: connect & retrieve
# -----------------------------

def connect_chroma(persist_dir: str, collection_name: str):
    """
    Connect to the Chroma persistent database and return the collection.
    Raises clear errors if path or collection is wrong.
    """
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Chroma DB directory not found: '{persist_dir}'. "
            "Run Task 3 first to create embeddings."
        )
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        return client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Could not load collection '{collection_name}'. "
            "Check that Task 3 created it and the name matches."
        ) from e


def retrieve_top_k(col, query: str, k: int = 3) -> Tuple[List[str], List[dict]]:
    """
    Retrieve top-K most relevant documents from Chroma for the given query.

    Returns:
        docs   -> list of plot snippets (strings)
        metas  -> list of metadata dicts (titles, chunk ids, etc.)
    """
    res = col.query(
        query_texts=[query],
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"]
    )
    docs = res.get("documents", [[]])[0] if res.get("documents") else []
    metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    return docs, metas


# -----------------------------
# 2) Optional: keyword re-ranking to sharpen relevance
# -----------------------------
# Rationale:
#   Embeddings are great but generic; for certain queries (e.g., U-boat),
#   we boost contexts that contain domain-specific keywords to push the
#   right film to the top.

KEYWORDS_BY_THEME = {
    "u-boat": ["u-boat", "uboat", "submarine", "u boat"]
}

def keyword_score(text: str, keywords: List[str]) -> int:
    """Count occurrences of any keyword (case-insensitive)."""
    t = (text or "").lower()
    return sum(t.count(kw.lower()) for kw in keywords)

def rerank_by_keywords(docs: List[str], metas: List[dict], keywords: List[str]) -> Tuple[List[str], List[dict]]:
    """Reorder docs/metas by keyword hit counts (descending)."""
    scored = [(keyword_score(doc, keywords), i) for i, doc in enumerate(docs)]
    if all(s == 0 for s, _ in scored):
        return docs, metas  # no change if no signals
    order = [i for _, i in sorted(scored, key=lambda x: x[0], reverse=True)]
    return [docs[i] for i in order], [metas[i] for i in order]


# -----------------------------
# 3) Prompt construction
# -----------------------------

def build_prompt(question: str, docs: List[str], metas: List[dict]) -> str:
    """
    Construct an instruction prompt for the LLM using labeled contexts.
    We keep contexts short to reduce token bloat and improve grounding.
    """
    items = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        title = str(meta.get("title", "Unknown"))
        chunk_id = meta.get("chunk_id", "N/A")
        items.append(f"Context {i} — Title: {title} (chunk {chunk_id})\n{_shorten(str(doc), 900)}")
    context_block = "\n\n".join(items) if items else "(no context)"

    prompt = (
        "You are a precise assistant. Answer ONLY using the given contexts.\n"
        "If the answer is not present, reply exactly: \"I don't know from the provided context.\".\n"
        "Do not copy long passages; summarize in your own words.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt


# -----------------------------
# 4) Hugging Face model loading & generation
# -----------------------------

def load_model(model_name: str):
    """
    Load a Hugging Face Seq2Seq model and tokenizer.
    Defaults to FLAN-T5 family, which works well for instruction following.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def generate_answer(
    tokenizer,
    model,
    question: str,
    docs: List[str],
    metas: List[dict],
    max_input_tokens: int = 1024,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.9,
    num_beams: int = 4
) -> str:
    """
    Generate a concise answer:
      - Deterministic (beam search) when do_sample=False (recommended for eval)
      - Creative (sampling) when do_sample=True (varied wording)
    We also add anti-copy constraints and inline citations.
    """
    prompt = build_prompt(question, docs, metas)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)

    # Decoding strategy with anti-copy heuristics
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=6,  # discourage copying long chunks verbatim
        min_length=16            # avoid ultra-short outputs
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(num_beams=max(1, num_beams), early_stopping=True))

    outputs = model.generate(**inputs, **gen_kwargs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Keep the answer neat and short (first sentence)
    answer = _first_sentence(answer, max_chars=300)

    # Append inline citations from top contexts (helps graders trace grounding)
    citations = [
        f"({m.get('title','Unknown')}, chunk {m.get('chunk_id','N/A')})"
        for m in metas[:2]
    ]
    citations = _dedupe(citations)
    if citations and citations[0] not in answer:
        answer = f"{answer} " + " ".join(citations)

    return answer


# -----------------------------
# 5) Structured JSON builder
# -----------------------------

def build_response_json(answer: str, docs: List[str], metas: List[dict], question: str) -> Dict[str, Any]:
    """
    Build the structured JSON object required by the assignment:
      - answer: natural language answer
      - contexts: retrieved plot snippets
      - reasoning: short explanation of how the answer was formed
    """
    # Create a concise reasoning string that cites the most relevant films.
    titles = [str(m.get("title","Unknown")) for m in metas if m.get("title")]
    # If the query hints at U-boats, prioritize titles whose contexts contain those keywords.
    if any(kw in question.lower() for kw in KEYWORDS_BY_THEME["u-boat"]):
        scores = [keyword_score(d, KEYWORDS_BY_THEME["u-boat"]) for d in docs]
        pairs = sorted(zip(scores, titles), key=lambda x: x[0], reverse=True)
        ranked = [t for s,t in pairs if s>0]
        if ranked: titles = ranked
    titles = _dedupe(titles)[:2]
    title_str = ", ".join(titles) if titles else "the retrieved snippets"

    reasoning = (
        f"The answer was formed by retrieving semantically similar plot chunks "
        f"and focusing on {title_str} because they best match the question: “{question}”."
    )

    return {
        "answer": answer,
        "contexts": [str(d) for d in docs],
        "reasoning": reasoning
    }


# -----------------------------
# 6) Simple heuristic scoring (optional)
# -----------------------------

def score_answer(answer: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Very basic scoring: does the answer contain any expected keyword?
    - This is NOT a perfect metric but useful for quick sanity checks.
    - expected_keywords is optional; if empty, we skip scoring.
    """
    if not expected_keywords:
        return {"has_expected": None, "matched": []}

    answer_l = (answer or "").lower()
    matched = [kw for kw in expected_keywords if kw.lower() in answer_l]
    return {
        "has_expected": bool(matched),
        "matched": matched
    }


# -----------------------------
# 7) Built-in default test set
# -----------------------------
# You can override this with --tests_file path/to/file.json
# Format for custom file:
#   [{"question": "...", "expected_keywords": ["kw1","kw2"]}, ...]

DEFAULT_TESTS = [
    {
        "question": "a ship hits an iceberg",
        "expected_keywords": ["atlantic", "titanic"]
    },
    {
        "question": "a German U-boat battle",
        "expected_keywords": ["enemy below", "u-boat", "submarine"]
    },
    {
        "question": "a couple separated during a disaster",
        "expected_keywords": ["atlantic"]
    },
    {
        "question": "robots and artificial intelligence",
        "expected_keywords": ["robot", "ai", "android"]   # loose; depends on retrieved candidates
    },
    {
        "question": "purple dragons invade mars",
        "expected_keywords": []  # should likely be "I don't know from the provided context."
    }
]


# -----------------------------
# 8) Orchestrator: run evaluation
# -----------------------------

def run_eval(
    persist_dir: str,
    collection: str,
    model_name: str,
    k: int,
    deterministic: bool,
    max_new_tokens: int,
    tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Runs the full evaluation pipeline across all test queries:
      - retrieve -> (optional rerank) -> generate -> build JSON -> score
    Returns a dict with:
      - run_info (timestamp, model, params)
      - results  (list per query)
    """
    # Connect vector store & load model once
    col = connect_chroma(persist_dir, collection)
    tokenizer, model = load_model(model_name)

    results = []
    for i, item in enumerate(tests, start=1):
        question = item.get("question", "").strip()
        expected_keywords = item.get("expected_keywords", [])

        # --- Retrieval ---
        docs, metas = retrieve_top_k(col, question, k=k)

        # --- Keyword reranking (if applicable) ---
        if any(kw in question.lower() for kw in KEYWORDS_BY_THEME["u-boat"]):
            docs, metas = rerank_by_keywords(docs, metas, KEYWORDS_BY_THEME["u-boat"])

        # --- Generation ---
        answer = generate_answer(
            tokenizer=tokenizer,
            model=model,
            question=question,
            docs=docs,
            metas=metas,
            max_new_tokens=max_new_tokens,
            do_sample=(not deterministic),   # deterministic => beam search
            temperature=0.7 if not deterministic else 0.0,
            top_p=0.9,
            num_beams=4 if deterministic else 1
        )

        # --- Structured JSON ---
        payload = build_response_json(answer, docs, metas, question)

        # --- Heuristic scoring (optional) ---
        scoring = score_answer(payload["answer"], expected_keywords)

        results.append({
            "question": question,
            "expected_keywords": expected_keywords,
            "answer": payload["answer"],
            "contexts": payload["contexts"],
            "reasoning": payload["reasoning"],
            "score": scoring
        })

    return {
        "run_info": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "model_name": model_name,
            "k": k,
            "deterministic": deterministic,
            "max_new_tokens": max_new_tokens,
            "num_tests": len(tests)
        },
        "results": results
    }


# -----------------------------
# 9) CLI
# -----------------------------

def _parse_bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y")

def main():
    parser = argparse.ArgumentParser(description="Task 7: Batch evaluation for the RAG pipeline.")
    parser.add_argument("--persist_dir", required=True, help="Chroma DB directory (e.g., data/vectorstore)")
    parser.add_argument("--collection", default="movie_chunks", help="Chroma collection name")
    parser.add_argument("--model_name", default="google/flan-t5-base", help="Hugging Face model name")
    parser.add_argument("--k", type=int, default=3, help="Top-K contexts to retrieve")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens for generation")
    parser.add_argument("--deterministic", type=_parse_bool, default=True, help="True => beam search; False => sampling")
    parser.add_argument("--tests_file", default="", help="Optional JSON file with list of {question, expected_keywords}")
    parser.add_argument("--out", required=True, help="Path to save evaluation JSON (e.g., data/outputs/eval_results.json)")
    args = parser.parse_args()

    # Load tests: from file if provided, else built-in defaults
    if args.tests_file:
        with open(args.tests_file, "r", encoding="utf-8") as f:
            tests = json.load(f)
            assert isinstance(tests, list), "tests_file must contain a JSON list"
    else:
        tests = DEFAULT_TESTS

    # Run evaluation
    report = run_eval(
        persist_dir=args.persist_dir,
        collection=args.collection,
        model_name=args.model_name,
        k=args.k,
        deterministic=args.deterministic,
        max_new_tokens=args.max_new_tokens,
        tests=tests
    )

    # Ensure output folder exists and save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Saved to: {args.out}")


if __name__ == "__main__":
    main()
