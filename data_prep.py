"""
data_prep.py — Format MTSamples CSV as a clinical note text corpus.

Usage:
    python data_prep.py                          # uses mtsamples.csv in current dir
    python data_prep.py --csv /path/to/mtsamples.csv
    python data_prep.py --output medical_train.txt

Input:  mtsamples.csv  (columns: description, medical_specialty, sample_name,
                                  transcription, keywords)
Output: medical_train.txt  (one document per entry, separated by blank lines)
"""

import argparse
import csv
import os
import sys


DOCUMENT_SEPARATOR = "\n\n"


def format_document(row: dict) -> str | None:
    """Format one MTSamples row as a structured clinical note text."""
    transcription = (row.get("transcription") or "").strip()
    if not transcription:
        return None  # skip rows with no transcription

    parts = []

    specialty = (row.get("medical_specialty") or "").strip()
    if specialty:
        parts.append(f"MEDICAL SPECIALTY: {specialty}")

    sample_name = (row.get("sample_name") or "").strip()
    if sample_name:
        parts.append(f"SAMPLE: {sample_name}")

    description = (row.get("description") or "").strip()
    if description:
        parts.append(f"DESCRIPTION: {description}")

    parts.append(f"TRANSCRIPTION:\n{transcription}")

    keywords = (row.get("keywords") or "").strip()
    if keywords:
        parts.append(f"KEYWORDS: {keywords}")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Format MTSamples for autoresearch")
    parser.add_argument("--csv", default="mtsamples.csv", help="Path to mtsamples.csv")
    parser.add_argument("--output", default="medical_train.txt", help="Output text file")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV file not found: {args.csv}")
        print("Download it first:")
        print("  kaggle datasets download -d tboyle10/medicaltranscriptions")
        print("  unzip medicaltranscriptions.zip")
        sys.exit(1)

    documents = []
    skipped = 0

    with open(args.csv, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = format_document(row)
            if doc:
                documents.append(doc)
            else:
                skipped += 1

    total_chars = sum(len(d) for d in documents)
    print(f"Documents: {len(documents)} kept, {skipped} skipped (no transcription)")
    print(f"Total characters: {total_chars:,} ({total_chars / 1e6:.1f}MB)")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(DOCUMENT_SEPARATOR.join(documents))
        f.write("\n")  # trailing newline

    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
