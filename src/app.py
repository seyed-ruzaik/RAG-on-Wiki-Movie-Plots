"""
src/app.py
==========
Task 6: Mini RAG Web App (Gradio) with Structured JSON Output

This app allows you to:
- Ask a question in the browser (Gradio UI).
- Retrieve the top-K relevant movie plot chunks from ChromaDB.
- Generate a natural language answer using a Hugging Face model.
- Output a structured JSON with:
    {
      "answer": "...",        # natural language answer
      "contexts": [...],      # the retrieved text snippets
      "reasoning": "..."      # short justification of how the answer was formed
    }
"""

import argparse
import json
import os
import re
from typing import List, Tuple

import chromadb                 # Vector database for retrieval
import gradio as gr             # Web UI library
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Hugging Face models


# -----------------------------
# Step 1: Connect to ChromaDB
# -----------------------------

def connect_chroma(persist_dir: str, collection_name: str):
    """
    Connect to the Chroma persistent database and return the collection.
    This is where your precomputed embeddings (Task 3) are stored.

    persist_dir: directory path where Chroma saved your DB
    collection_name: the collection name used during indexing
    """
    if not os.path.isdir(persist_dir):
        raise FileNotFoundError(
            f"Chroma DB directory not found: '{persist_dir}'. "
            "Run Task 3 first to create embeddings."
        )
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(collection_name)


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
# Step 2: Helper functions
# -----------------------------

def _shorten(text: str, max_chars: int = 900) -> str:
    """Limit long text chunks to avoid overloading the prompt."""
    t = (text or "").strip()
    return (t[:max_chars] + " ...") if len(t) > max_chars else t

def _first_sentence(text: str, max_chars: int = 300) -> str:
    """Keep only the first sentence (or up to max_chars) to make answers concise."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    s = parts[0] if parts and parts[0] else text[:max_chars]
    return s[:max_chars].strip()

def _dedupe(seq: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# -----------------------------
# Step 3: Keyword-based reranking
# -----------------------------
# Problem: sometimes the retriever brings partially related films.
# Fix: we boost relevance manually if the question mentions special keywords.

KEYWORDS_BY_THEME = {
    "u-boat": ["u-boat", "uboat", "submarine", "u boat"]
}

def keyword_score(text: str, keywords: list[str]) -> int:
    """Count keyword occurrences in a text."""
    t = (text or "").lower()
    return sum(t.count(kw.lower()) for kw in keywords)

def rerank_by_keywords(docs, metas, keywords):
    """
    Reorder documents by keyword hits.
    Example: If query mentions 'u-boat', then contexts with 'u-boat' will rank higher.
    """
    scored = [(keyword_score(doc, keywords), i) for i, doc in enumerate(docs)]
    if all(s == 0 for s, _ in scored):
        return docs, metas  # nothing to change
    order = [i for _, i in sorted(scored, key=lambda x: x[0], reverse=True)]
    return [docs[i] for i in order], [metas[i] for i in order]


# -----------------------------
# Step 4: Prompt & Answer generation
# -----------------------------

def build_prompt(question: str, docs: List[str], metas: List[dict]) -> str:
    """
    Construct the input prompt for the LLM.
    It includes the retrieved contexts and the question.
    """
    items = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        title = str(meta.get("title", "Unknown"))
        chunk_id = meta.get("chunk_id", "N/A")
        items.append(f"Context {i} — Title: {title} (chunk {chunk_id})\n{_shorten(str(doc), 900)}")
    context_block = "\n\n".join(items) if items else "(no context)"
    return (
        "You are a precise assistant. Answer ONLY using the given contexts.\n"
        "If the answer is not present, reply exactly: \"I don't know from the provided context.\".\n"
        "Do not copy long passages; summarize in your own words.\n\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def load_model(model_name: str):
    """
    Load the Hugging Face model + tokenizer.
    By default we use a T5-family model (flan-t5-small/base).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def generate_answer(tokenizer, model, question, docs, metas,
                    max_input_tokens=1024, max_new_tokens=128,
                    do_sample=False, temperature=0.0, top_p=0.9, num_beams=4) -> str:
    """
    Generate an answer using the LLM.

    - deterministic (beam search) if do_sample=False
    - creative (sampling) if do_sample=True
    """
    prompt = build_prompt(question, docs, metas)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)

    # Decoding settings
    gen_kwargs = dict(max_new_tokens=max_new_tokens, no_repeat_ngram_size=6, min_length=16)
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))
    else:
        gen_kwargs.update(dict(num_beams=num_beams, early_stopping=True))

    # Generate
    outputs = model.generate(**inputs, **gen_kwargs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Post-process to keep answer short and readable
    answer = _first_sentence(answer, max_chars=300)

    # Add inline citations (from top 1–2 contexts)
    citations = [f"({m.get('title','Unknown')}, chunk {m.get('chunk_id','N/A')})" for m in metas[:2]]
    citations = _dedupe(citations)
    if citations and citations[0] not in answer:
        answer = f"{answer} " + " ".join(citations)

    return answer


# -----------------------------
# Step 5: Build structured JSON
# -----------------------------

def build_response_json(answer, docs, metas, question) -> str:
    """
    Build the JSON output required by the assignment:
    {
      "answer": "...",
      "contexts": [...],
      "reasoning": "..."
    }
    """
    # Get titles from metadata
    titles = [str(m.get("title","Unknown")) for m in metas if m.get("title")]

    # If question is about U-boats, prioritize those titles
    if any(kw in question.lower() for kw in KEYWORDS_BY_THEME["u-boat"]):
        scores = [keyword_score(d, KEYWORDS_BY_THEME["u-boat"]) for d in docs]
        pairs = sorted(zip(scores, titles), key=lambda x: x[0], reverse=True)
        ranked = [t for s,t in pairs if s>0]
        if ranked: titles = ranked

    # Deduplicate & shorten to 2 titles max
    titles = _dedupe(titles)[:2]
    title_str = ", ".join(titles) if titles else "the retrieved snippets"

    reasoning = (
        f"The answer was formed by retrieving semantically similar plot chunks "
        f"and focusing on {title_str} because they best match the question: “{question}”."
    )

    return json.dumps({"answer": answer, "contexts": [str(d) for d in docs], "reasoning": reasoning},
                      indent=2, ensure_ascii=False)


# -----------------------------
# Step 6: Gradio web app
# -----------------------------

def build_interface(col, tokenizer, model, k: int):
    """
    Build the Gradio web interface.
    It shows:
      - Answer
      - Retrieved contexts
      - Structured JSON
    """
    def rag_answer(question: str, max_new_tokens: int, deterministic: bool):
        if not question.strip():
            return "Please enter a question.", "", "{}"

        # 1. Retrieve
        docs, metas = retrieve_top_k(col, question, k=k)

        # 2. Rerank if query mentions special keywords (like "u-boat")
        if any(kw in question.lower() for kw in KEYWORDS_BY_THEME["u-boat"]):
            docs, metas = rerank_by_keywords(docs, metas, KEYWORDS_BY_THEME["u-boat"])

        # 3. Generate answer
        answer = generate_answer(tokenizer, model, question, docs, metas,
                                 max_new_tokens=max_new_tokens,
                                 do_sample=(not deterministic))

        # 4. Format contexts for display
        contexts_md = "\n---\n".join([
            f"**{m.get('title','Unknown')} (chunk {m.get('chunk_id','N/A')})**\n{_shorten(d,1500)}"
            for d,m in zip(docs,metas)
        ])

        # 5. Build JSON
        json_str = build_response_json(answer, docs, metas, question)

        return answer, contexts_md, json_str

    with gr.Blocks(title="Mini RAG App") as demo:
        gr.Markdown("## 🎬 Wiki Movie Plots — RAG Demo\nAsk a question, see retrieved contexts, and structured JSON.")
        q = gr.Textbox(label="Your question", placeholder="e.g., a ship hits an iceberg")
        deterministic = gr.Checkbox(value=True, label="Deterministic (beam search)")
        max_new = gr.Slider(32, 256, value=128, step=16, label="Max new tokens")
        btn = gr.Button("Generate Answer")
        ans = gr.Markdown(label="Answer")
        ctxs = gr.Markdown(label="Retrieved Contexts")
        jbox = gr.Code(label="Structured JSON", language="json")
        btn.click(fn=rag_answer, inputs=[q, max_new, deterministic], outputs=[ans, ctxs, jbox])
    return demo


# -----------------------------
# Step 7: Main entrypoint
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", required=True, help="Chroma DB directory")
    parser.add_argument("--collection", default="movie_chunks", help="Chroma collection name")
    parser.add_argument("--k", type=int, default=3, help="Top-K contexts to retrieve")
    parser.add_argument("--model_name", default="google/flan-t5-base", help="HuggingFace model")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port")
    args = parser.parse_args()

    # Connect DB + load model
    col = connect_chroma(args.persist_dir, args.collection)
    tokenizer, model = load_model(args.model_name)

    # Launch Gradio app
    demo = build_interface(col, tokenizer, model, k=args.k)
    demo.queue()
    demo.launch(server_port=args.server_port)


if __name__ == "__main__":
    main()
