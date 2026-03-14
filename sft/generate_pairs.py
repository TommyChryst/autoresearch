"""
generate_pairs.py — Generate (prompt, SOAP note) pairs from MTSamples using GPT-4o-mini.

Usage:
    python sft/generate_pairs.py --dry-run           # print 3 prompts, no API calls
    python sft/generate_pairs.py --count 10          # test batch, ~$0.01
    python sft/generate_pairs.py --count 500         # full run, ~$0.30
    python sft/generate_pairs.py --count 500 --resume  # resume if interrupted

Requires:
    OPENAI_API_KEY env var
    pip install openai pandas

Output:
    sft/pairs.jsonl — one {"prompt": "...", "response": "..."} per line
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = REPO_ROOT / "mtsamples.csv"
PAIRS_FILE = SCRIPT_DIR / "pairs.jsonl"
PROGRESS_FILE = SCRIPT_DIR / "pairs_progress.jsonl"

MODEL = "gpt-4o-mini"
MAX_CONCURRENT = 5
MAX_RETRIES = 5
BASE_BACKOFF = 2.0

MIN_TRANSCRIPTION_CHARS = 200
MAX_PER_SPECIALTY = 60

SKIP_SPECIALTIES = {
    "Radiology",
    "Letters",
    "Autopsy",
    "Lab Medicine - Pathology",
    "Discharge Summary",
    "IME-QME-Work Comp etc.",
    "Speech - Language",
    "Hospice - Palliative Care",
}

GENERATION_SYSTEM = """\
You are an expert clinical documentation specialist with 20 years of experience writing \
hospital-grade SOAP notes. You write terse, precise, professional clinical documentation.

SOAP NOTE FORMAT - follow this EXACTLY, no additions:

**SUBJECTIVE:**
CC: [chief complaint in patient's own words, 1 line]
HPI: [onset, provocation/palliation, quality, radiation, severity /10, timing, context]
PMH: [relevant past medical history]
Meds: [current medications with doses]
Allergies: [drug allergies or NKDA]
SH: [smoking/alcohol/drugs, occupation, living situation - brief]
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
1. [Primary diagnosis/working diagnosis] - [brief rationale]
DDx: [differential diagnoses in order of likelihood]

**PLAN:**
1. [Diagnostic orders - specific tests]
2. [Therapeutic interventions - specific medications/doses]
3. [Consults if needed]
4. [Patient education - 1 line]
5. Follow-up: [timeframe and conditions for return]

RULES:
- Use standard medical abbreviations (HTN, DM2, SOB, CP, MI, ACS, ACE inhibitor, etc.)
- Be concise - bullet points, not paragraphs
- Vital signs are REQUIRED in every Objective section (estimate if not provided)
- Physical exam is REQUIRED (at minimum CV and relevant systems)
- Differential diagnosis is REQUIRED in Assessment
- DO NOT add sections beyond S/O/A/P
- Drug class accuracy is critical: lisinopril is an ACE inhibitor, metoprolol is a beta-blocker, etc.

---
TASK:
Given a real medical transcription, produce a JSON object with two fields:
1. "prompt": A 1-3 sentence clinical scenario written naturally, as a clinician would describe
   the patient. DO NOT copy the transcription verbatim - synthesize key clinical facts
   (age, sex, chief complaint, relevant PMH, key findings) into a natural request for a SOAP note.
2. "response": A complete SOAP note following the format above, reformatted from the transcription.
   All four sections (SUBJECTIVE, OBJECTIVE, ASSESSMENT, PLAN) are required.
   Estimate vitals and physical exam if not explicitly stated.

Return ONLY valid JSON. No extra text, no markdown code fences.
Example: {"prompt": "62yo male presents with acute chest pain...", "response": "**SUBJECTIVE:**\\nCC: ..."}\
"""


# ---------------------------------------------------------------------------
# CSV loading and filtering
# ---------------------------------------------------------------------------
def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], count: int) -> list[dict]:
    specialty_counts: dict[str, int] = {}
    filtered = []
    for row in rows:
        specialty = (row.get("medical_specialty") or "").strip()
        transcription = (row.get("transcription") or "").strip()
        if specialty in SKIP_SPECIALTIES:
            continue
        if len(transcription) < MIN_TRANSCRIPTION_CHARS:
            continue
        current = specialty_counts.get(specialty, 0)
        if current >= MAX_PER_SPECIALTY:
            continue
        specialty_counts[specialty] = current + 1
        filtered.append(row)
    random.shuffle(filtered)
    return filtered[:count]


def build_user_message(row: dict) -> str:
    specialty = (row.get("medical_specialty") or "").strip()
    sample_name = (row.get("sample_name") or "").strip()
    description = (row.get("description") or "").strip()
    transcription = (row.get("transcription") or "").strip()
    parts = []
    if specialty:
        parts.append(f"SPECIALTY: {specialty}")
    if sample_name:
        parts.append(f"CASE: {sample_name}")
    if description:
        parts.append(f"DESCRIPTION: {description}")
    parts.append(f"TRANSCRIPTION:\n{transcription[:3000]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------
async def call_gpt(client, row: dict, sem: asyncio.Semaphore) -> dict | None:
    user_msg = build_user_message(row)
    backoff = BASE_BACKOFF
    loop = asyncio.get_event_loop()

    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                response = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=MODEL,
                        max_tokens=1500,
                        messages=[
                            {"role": "system", "content": GENERATION_SYSTEM},
                            {"role": "user", "content": user_msg},
                        ],
                        response_format={"type": "json_object"},
                    ),
                )
            text = response.choices[0].message.content.strip()
            pair = json.loads(text)
            if "prompt" not in pair or "response" not in pair:
                raise ValueError("Missing required keys")
            return pair

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            is_last = attempt == MAX_RETRIES - 1
            if is_rate_limit:
                wait = backoff * (2 ** attempt)
                print(f"  Rate limited, waiting {wait:.0f}s...", flush=True)
                await asyncio.sleep(wait)
            elif is_last:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}", flush=True)
                return None
            else:
                await asyncio.sleep(backoff)

    return None


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
def dry_run(rows: list[dict], n: int = 3):
    print(f"=== DRY RUN - showing {n} formatted prompts (no API calls) ===\n")
    for i, row in enumerate(rows[:n], 1):
        print(f"--- Row {i} ---")
        print(build_user_message(row)[:500])
        print("...\n")


# ---------------------------------------------------------------------------
# Main async loop
# ---------------------------------------------------------------------------
async def run(args):
    from openai import OpenAI

    rows = load_rows(args.csv)
    print(f"Loaded {len(rows)} total rows from {args.csv}")

    target_rows = filter_rows(rows, args.count)
    print(f"After filtering: {len(target_rows)} rows eligible")

    if args.dry_run:
        dry_run(target_rows)
        return

    completed_pairs: list[dict] = []
    if args.resume and PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_pairs.append(json.loads(line))
        done_count = len(completed_pairs)
        print(f"Resuming: {done_count} pairs already done")
        target_rows = target_rows[done_count:]

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    progress_fh = open(PROGRESS_FILE, "a", encoding="utf-8")
    total = len(target_rows)
    success = 0
    fail = 0

    print(f"Generating {total} pairs with {MODEL}...")
    print(f"Estimated cost: ~${total * 0.0006:.2f}\n")

    for i, row in enumerate(target_rows):
        specialty = (row.get("medical_specialty") or "Unknown").strip()
        print(f"[{i+1}/{total}] {specialty[:40]}", end="  ", flush=True)

        pair = await call_gpt(client, row, sem)
        if pair:
            line = json.dumps(pair)
            progress_fh.write(line + "\n")
            progress_fh.flush()
            completed_pairs.append(pair)
            success += 1
            print(f"OK ({len(pair['prompt'])}p / {len(pair['response'])}r chars)")
        else:
            fail += 1
            print("SKIP")

    progress_fh.close()

    with open(PAIRS_FILE, "w", encoding="utf-8") as f:
        for pair in completed_pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\nDone: {success} pairs written to {PAIRS_FILE}")
    print(f"Skipped: {fail} rows")
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate SFT pairs from MTSamples via GPT-4o-mini")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if not args.dry_run and "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY env var not set")
        sys.exit(1)

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
