"""
prepare_medical.py — Data preparation for autoresearch using MTSamples medical data.

Replaces the default prepare.py data pipeline (FineWeb parquet downloads) with a
local medical text corpus. After running this script, the cache directory contains
parquet shards and a trained tokenizer — train.py (which imports from prepare.py)
will work as-is.

Usage:
    python prepare_medical.py                          # uses medical_train.txt
    python prepare_medical.py --txt /path/to/medical_train.txt

Steps:
    1. Read medical_train.txt and split into documents
    2. Split 90/10 train/val
    3. Write parquet shards to ~/.cache/autoresearch/data/
    4. Train BPE tokenizer on training documents
    5. Save tokenizer + token_bytes to ~/.cache/autoresearch/tokenizer/
"""

import argparse
import math
import os
import pickle
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Mirror the constants from prepare.py exactly so train.py works unchanged
# ---------------------------------------------------------------------------

CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR      = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

VAL_SHARD    = 6542
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE   = 8192

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN      = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Step 1 — Read + split documents
# ---------------------------------------------------------------------------

def read_documents(txt_path: str) -> list[str]:
    print(f"Reading documents from: {txt_path}")
    with open(txt_path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Documents are separated by double newlines (as written by data_prep.py)
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    print(f"  Found {len(docs):,} documents")
    total_chars = sum(len(d) for d in docs)
    print(f"  Total characters: {total_chars:,} ({total_chars / 1e6:.2f} MB)")
    return docs


def split_train_val(docs: list[str], val_fraction: float = 0.10):
    """Use last val_fraction of docs as validation, rest as training."""
    n_val = max(1, int(len(docs) * val_fraction))
    train = docs[:-n_val]
    val   = docs[-n_val:]
    print(f"  Train: {len(train):,} docs | Val: {len(val):,} docs")
    return train, val


# ---------------------------------------------------------------------------
# Step 2 — Write parquet shards
# ---------------------------------------------------------------------------

def write_parquet_shards(docs: list[str], shard_prefix: str, docs_per_shard: int, start_index: int = 0):
    """Write docs to numbered parquet files. Each file has a 'text' column."""
    os.makedirs(DATA_DIR, exist_ok=True)
    n_shards = math.ceil(len(docs) / docs_per_shard)
    paths = []
    for i in range(n_shards):
        batch = docs[i * docs_per_shard : (i + 1) * docs_per_shard]
        shard_idx = start_index + i
        filename = f"shard_{shard_idx:05d}.parquet"
        filepath = os.path.join(DATA_DIR, filename)
        table = pa.table({"text": pa.array(batch, type=pa.string())})
        pq.write_table(table, filepath)
        paths.append(filepath)
        print(f"  Wrote {filename} ({len(batch)} docs)")
    return paths


def write_val_shard(val_docs: list[str]):
    """Write the fixed val shard (shard_06542.parquet)."""
    filepath = os.path.join(DATA_DIR, VAL_FILENAME)
    table = pa.table({"text": pa.array(val_docs, type=pa.string())})
    pq.write_table(table, filepath)
    print(f"  Wrote {VAL_FILENAME} ({len(val_docs)} docs) [val]")
    return filepath


# ---------------------------------------------------------------------------
# Step 3 — Train tokenizer (mirrors prepare.py logic)
# ---------------------------------------------------------------------------

def train_tokenizer(train_docs: list[str]):
    tokenizer_pkl    = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}, skipping.")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    print("Tokenizer: training BPE tokenizer on medical corpus...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)

    # Train from the list of training documents
    tokenizer.train_from_iterator(iter(train_docs), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding
    pattern          = tokenizer.get_pattern()
    mergeable_ranks  = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset    = len(mergeable_ranks)
    special_tokens_d = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens_d,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s (vocab_size={enc.n_vocab})")

    # Build token_bytes lookup
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test    = "Hello world! SOAP note: Subjective — patient reports pain."
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print("Tokenizer: sanity check passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare medical data for autoresearch")
    parser.add_argument("--txt", default="medical_train.txt",
                        help="Path to medical_train.txt (created by data_prep.py)")
    parser.add_argument("--docs-per-shard", type=int, default=500,
                        help="Documents per training parquet shard")
    parser.add_argument("--val-fraction", type=float, default=0.10,
                        help="Fraction of documents to use for validation")
    parser.add_argument("--retrain-tokenizer", action="store_true",
                        help="Force retrain even if tokenizer already exists")
    args = parser.parse_args()

    if not os.path.exists(args.txt):
        print(f"ERROR: Text file not found: {args.txt}")
        print("Run data_prep.py first:")
        print("  python data_prep.py --csv mtsamples.csv --output medical_train.txt")
        sys.exit(1)

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Read documents
    docs = read_documents(args.txt)
    if len(docs) < 10:
        print("ERROR: Too few documents found. Check medical_train.txt.")
        sys.exit(1)

    train_docs, val_docs = split_train_val(docs, val_fraction=args.val_fraction)
    print()

    # Step 2: Write parquet shards
    print("Writing parquet shards...")
    write_parquet_shards(train_docs, "shard", docs_per_shard=args.docs_per_shard, start_index=0)
    write_val_shard(val_docs)
    print()

    # Step 3: Train tokenizer
    if args.retrain_tokenizer:
        # Remove old tokenizer files to force retrain
        for fname in ["tokenizer.pkl", "token_bytes.pt"]:
            p = os.path.join(TOKENIZER_DIR, fname)
            if os.path.exists(p):
                os.remove(p)
                print(f"Removed {p}")

    train_tokenizer(train_docs)
    print()
    print("Done! Ready to train.")
    print()
    print("Next steps:")
    print("  uv run train.py              # baseline test run")
    print("  grep '^val_bpb:' run.log     # after redirecting output")


if __name__ == "__main__":
    main()
