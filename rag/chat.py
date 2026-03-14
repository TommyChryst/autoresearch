"""
chat.py — RAG-grounded medical chatbot via Ollama.

Usage:
    python rag/chat.py
    python rag/chat.py --model qwen2.5:3b
    python rag/chat.py --specialty Cardiology
    python rag/chat.py --top-k 8 --no-stream

Requires:
    - Ollama running: ollama serve  (or it starts automatically on Windows)
    - Model pulled:   ollama pull phi3.5
    - Chroma DB built: python rag/embed_mtsamples.py

Dependencies:
    pip install chromadb sentence-transformers ollama
"""

import argparse
import os
import sys
import textwrap
from typing import Iterator

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "medical-soap"
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "chroma_db")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mtsamples"
TOP_K = 5
MAX_HISTORY_TURNS = 6   # how many prior turns to keep in context
MAX_CHUNK_CHARS = 800   # max chars per retrieved chunk in the prompt

SYSTEM_PROMPT = """\
You are an expert clinical documentation specialist with 20 years of experience writing \
hospital-grade SOAP notes. You write terse, precise, professional clinical documentation.

SOAP NOTE FORMAT — follow this EXACTLY, no additions:

**SUBJECTIVE:**
CC: [chief complaint in patient's own words, 1 line]
HPI: [onset, provocation/palliation, quality, radiation, severity /10, timing, context]
PMH: [relevant past medical history]
Meds: [current medications with doses]
Allergies: [drug allergies or NKDA]
SH: [smoking/alcohol/drugs, occupation, living situation — brief]
FH: [relevant family history]
ROS: [pertinent positives and negatives only]

**OBJECTIVE:**
Vitals: BP _/_ | HR _ | RR _ | Temp _ | SpO2 _%
General: [general appearance]
CV: [heart sounds, rhythm, murmurs, peripheral pulses, edema]
Resp: [breath sounds, effort]
Abd: [if relevant]
[other relevant exam systems]

**ASSESSMENT:**
1. [Primary diagnosis/working diagnosis] — [brief rationale]
DDx: [differential diagnoses in order of likelihood]

**PLAN:**
1. [Diagnostic orders — specific tests]
2. [Therapeutic interventions — specific medications/doses]
3. [Consults if needed]
4. [Patient education — 1 line]
5. Follow-up: [timeframe and conditions for return]

RULES:
- Use standard medical abbreviations (HTN, DM2, SOB, CP, MI, ACS, ACE inhibitor, etc.)
- Be concise — bullet points, not paragraphs
- Vital signs are REQUIRED in every Objective section (estimate if not provided)
- Physical exam is REQUIRED (at minimum CV and relevant systems)
- Differential diagnosis is REQUIRED in Assessment
- DO NOT add sections beyond S/O/A/P (no "Patient's Response", "Acknowledgment", "Documentation", etc.)
- DO NOT copy patient demographics from the examples — examples show STYLE only
- The examples below are real MTSamples notes — use them for phrasing/format, not patient facts
- Drug class accuracy is critical: lisinopril is an ACE inhibitor, metoprolol is a beta-blocker, etc.\
"""


# ---------------------------------------------------------------------------
# Chroma retrieval
# ---------------------------------------------------------------------------
def load_collection(db_path: str, embed_model: str):
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        print("ERROR: Run: pip install chromadb sentence-transformers")
        sys.exit(1)

    if not os.path.exists(db_path):
        print(f"ERROR: Chroma DB not found at {db_path}")
        print("Run first: python rag/embed_mtsamples.py")
        sys.exit(1)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except Exception:
        print(f"ERROR: Collection '{COLLECTION_NAME}' not found in {db_path}")
        print("Run: python rag/embed_mtsamples.py")
        sys.exit(1)

    return collection


def retrieve(collection, query: str, top_k: int, specialty_filter: str | None) -> list[dict]:
    where = None
    if specialty_filter:
        # Chroma where filter — partial match not supported, use exact specialty name
        where = {"specialty": {"$eq": specialty_filter}}

    kwargs = dict(query_texts=[query], n_results=top_k)
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        # If specialty filter returns 0 results, fall back without filter
        if specialty_filter:
            print(f"  [Note: specialty filter '{specialty_filter}' returned no results, searching all]")
            results = collection.query(query_texts=[query], n_results=top_k)
        else:
            raise e

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "meta": meta, "distance": dist})
    return chunks


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(
    user_query: str,
    retrieved_chunks: list[dict],
    history: list[dict],
) -> list[dict]:
    """Build the messages list for Ollama chat API."""

    # Format retrieved examples
    examples_text = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["meta"]
        header = f"[Example {i} — {meta.get('specialty', 'Unknown')} | {meta.get('sample_name', '')}]"
        body = chunk["text"][:MAX_CHUNK_CHARS]
        if len(chunk["text"]) > MAX_CHUNK_CHARS:
            body += "..."
        examples_text += f"\n{header}\n{body}\n"

    # Build system message with injected examples
    system_with_examples = SYSTEM_PROMPT
    if examples_text:
        system_with_examples += f"\n\n---\nRELEVANT MTSamples EXAMPLES:\n{examples_text}---"

    messages = [{"role": "system", "content": system_with_examples}]

    # Add conversation history (last N turns)
    messages.extend(history[-MAX_HISTORY_TURNS * 2:])

    # Add current user message
    messages.append({"role": "user", "content": user_query})

    return messages


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------
def check_ollama(model: str):
    try:
        import ollama
    except ImportError:
        print("ERROR: Run: pip install ollama")
        sys.exit(1)

    try:
        ollama.show(model)
    except Exception:
        print(f"Model '{model}' not found locally.")
        print(f"Pull it with: ollama pull {model}")
        print("If Ollama isn't installed: https://ollama.com/download")
        sys.exit(1)


def stream_response(model: str, messages: list[dict]) -> Iterator[str]:
    import ollama
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk["message"]["content"]
        if delta:
            yield delta


def get_response(model: str, messages: list[dict]) -> str:
    import ollama
    response = ollama.chat(model=model, messages=messages, stream=False)
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# CLI chat loop
# ---------------------------------------------------------------------------
def print_sources(chunks: list[dict]):
    print("\n  \033[90m[Sources retrieved:]")
    for i, c in enumerate(chunks, 1):
        meta = c["meta"]
        dist = c["distance"]
        print(f"  {i}. {meta.get('specialty', '?')} | {meta.get('sample_name', '?')} (dist={dist:.3f})")
    print("\033[0m", end="")


def run_chat(args):
    print(f"\033[1mMedical RAG Chatbot\033[0m")
    print(f"Model: {args.model}  |  Embedding: {args.embed_model}  |  Top-K: {args.top_k}")
    if args.specialty:
        print(f"Specialty filter: {args.specialty}")
    print("Type 'quit' or Ctrl+C to exit. Type '/sources' to toggle source display.\n")

    # Load resources
    print("Loading Chroma DB...", end=" ", flush=True)
    collection = load_collection(args.db, args.embed_model)
    print(f"OK ({collection.count()} chunks)")

    print(f"Checking Ollama model '{args.model}'...", end=" ", flush=True)
    check_ollama(args.model)
    print("OK\n")

    history: list[dict] = []
    show_sources = args.show_sources

    try:
        while True:
            try:
                user_input = input("\033[32mYou:\033[0m ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "/sources":
                show_sources = not show_sources
                print(f"  [Source display: {'ON' if show_sources else 'OFF'}]")
                continue
            if user_input.lower() == "/clear":
                history = []
                print("  [Conversation history cleared]")
                continue
            if user_input.lower() == "/help":
                print("  Commands: /sources (toggle), /clear (reset history), quit")
                continue

            # Retrieve relevant chunks
            chunks = retrieve(collection, user_input, args.top_k, args.specialty)

            if show_sources:
                print_sources(chunks)

            # Build messages and get response
            messages = build_prompt(user_input, chunks, history)

            print("\n\033[34mAssistant:\033[0m ", end="", flush=True)

            if args.stream:
                full_response = ""
                for token in stream_response(args.model, messages):
                    print(token, end="", flush=True)
                    full_response += token
                print("\n")
            else:
                full_response = get_response(args.model, messages)
                # Wrap for readability
                for line in full_response.split("\n"):
                    if len(line) > 100:
                        print(textwrap.fill(line, width=100))
                    else:
                        print(line)
                print()

            # Update history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        pass

    print("\nGoodbye.")


def main():
    parser = argparse.ArgumentParser(description="RAG medical chatbot via Ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--db", default=DEFAULT_DB, help="Chroma DB path")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Sentence-transformers model")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--specialty", default=None, help="Filter by medical specialty (exact match)")
    parser.add_argument("--no-stream", dest="stream", action="store_false", help="Disable streaming output")
    parser.add_argument("--show-sources", action="store_true", help="Show retrieved sources by default")
    parser.set_defaults(stream=True)
    args = parser.parse_args()
    run_chat(args)


if __name__ == "__main__":
    main()
