"""
embed_mtsamples.py — Index MTSamples transcriptions into a local Chroma vector store.

Usage:
    python rag/embed_mtsamples.py
    python rag/embed_mtsamples.py --csv /path/to/mtsamples.csv
    python rag/embed_mtsamples.py --model pubmedbert-base-embeddings  # better medical retrieval

Output:
    rag/chroma_db/   (local persistent Chroma collection)

Requires:
    pip install chromadb sentence-transformers pandas
"""

import argparse
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "mtsamples.csv")
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "mtsamples"
CHUNK_SIZE = 400        # tokens (approximate — we split by whitespace words)
CHUNK_OVERLAP = 50      # words of overlap between chunks
BATCH_SIZE = 64         # how many chunks to embed at once


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-count-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def load_mtsamples(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    before = len(df)
    df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]
    print(f"Loaded {len(df)} rows with transcriptions (dropped {before - len(df)} empty)")
    return df


def build_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict], list[str]]:
    """Return (texts, metadatas, ids) ready for Chroma insertion."""
    texts, metadatas, ids = [], [], []
    for i, row in df.iterrows():
        transcription = str(row.get("transcription", "")).strip()
        specialty = str(row.get("medical_specialty", "")).strip()
        sample_name = str(row.get("sample_name", "")).strip()
        description = str(row.get("description", "")).strip()
        keywords = str(row.get("keywords", "")).strip()

        doc_chunks = chunk_text(transcription)
        for j, chunk in enumerate(doc_chunks):
            chunk_id = f"doc_{i}_chunk_{j}"
            texts.append(chunk)
            metadatas.append({
                "specialty": specialty,
                "sample_name": sample_name,
                "description": description,
                "keywords": keywords,
                "doc_index": int(i),
                "chunk_index": int(j),
            })
            ids.append(chunk_id)

    return texts, metadatas, ids


def index(csv_path: str, db_path: str, embedding_model: str, reset: bool = False):
    # ---- Imports (late so errors are clear) ----
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb sentence-transformers")
        sys.exit(1)

    print(f"Loading MTSamples from: {csv_path}")
    df = load_mtsamples(csv_path)

    print(f"Building chunks (size={CHUNK_SIZE} words, overlap={CHUNK_OVERLAP})...")
    texts, metadatas, ids = build_chunks(df)
    print(f"Total chunks: {len(texts)}")

    print(f"Loading embedding model: {embedding_model}")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    print(f"Opening Chroma DB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    if reset and COLLECTION_NAME in [c.name for c in client.list_collections()]:
        print(f"Resetting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not reset:
        print(f"Collection already has {existing} vectors. Use --reset to re-index.")
        return collection

    # Insert in batches
    print(f"Embedding and indexing {len(texts)} chunks in batches of {BATCH_SIZE}...")
    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        batch_texts = texts[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]
        collection.add(documents=batch_texts, metadatas=batch_meta, ids=batch_ids)
        pct = end / len(texts) * 100
        print(f"  [{end}/{len(texts)}] {pct:.0f}%", end="\r")

    print(f"\nIndexed {collection.count()} chunks into '{COLLECTION_NAME}'.")
    print(f"DB saved to: {os.path.abspath(db_path)}")
    return collection


def test_retrieval(db_path: str, embedding_model: str):
    """Quick smoke-test: query a few things and print results."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    queries = [
        "chest pain cardiology hypertension",
        "knee replacement orthopedic surgery",
        "SOAP note subjective objective assessment plan",
    ]
    for q in queries:
        results = collection.query(query_texts=[q], n_results=3)
        print(f"\nQuery: '{q}'")
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta['specialty']}] {meta['sample_name']}: {doc[:100]}...")


def main():
    parser = argparse.ArgumentParser(description="Index MTSamples into Chroma")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to mtsamples.csv")
    parser.add_argument("--db", default=DEFAULT_DB, help="Chroma DB output directory")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model name",
    )
    parser.add_argument("--reset", action="store_true", help="Delete and re-index existing DB")
    parser.add_argument("--test", action="store_true", help="Run retrieval smoke-test after indexing")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: mtsamples.csv not found at: {args.csv}")
        print("Download it:")
        print("  kaggle datasets download -d tboyle10/medicaltranscriptions")
        print("  unzip medicaltranscriptions.zip -d .")
        print("  # or visit: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        sys.exit(1)

    index(args.csv, args.db, args.model, reset=args.reset)

    if args.test:
        print("\n--- Retrieval smoke-test ---")
        test_retrieval(args.db, args.model)


if __name__ == "__main__":
    main()
