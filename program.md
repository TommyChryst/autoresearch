# autoresearch — Healthcare SOAP Note Summarization

This is an autonomous experiment to train a small, specialized LLM for clinical
documentation — specifically, compressing and summarizing medical transcriptions
into SOAP-format (Subjective, Objective, Assessment, Plan) clinical notes.

## Dataset

The model trains on **MTSamples** (~4,000 real medical transcriptions spanning 40+
specialties). Each document is a structured clinical note with fields:

- `MEDICAL SPECIALTY` — e.g. Cardiology, Orthopedics, Psychiatry
- `SAMPLE` — procedure or encounter name
- `DESCRIPTION` — brief summary
- `TRANSCRIPTION` — full dictated clinical note
- `KEYWORDS` — relevant medical terms

The corpus is small (~10 MB of text). This means:
- Overfitting risk is real — watch val_bpb carefully vs. train loss
- The model can cycle through the training set many times per run
- Regularization (weight decay, dropout-equivalent) may matter more here than on
  large-scale web text

## Domain-Specific Considerations

**Vocabulary**: Medical terminology is dense and highly structured. The custom
BPE tokenizer (vocab_size=8192) trained on this corpus should encode clinical
terms more efficiently than a general tokenizer. This is an advantage — exploit it.

**Document structure**: Documents are short (avg ~500-1000 tokens) and highly
structured. This is unlike web text. Consider:
- Whether short context windows might outperform long ones given short doc length
- Whether a smaller model might generalize better on a small domain corpus
- Positional encoding behavior on short vs. long sequences

**Evaluation metric**: val_bpb measures compression of held-out clinical notes.
Lower val_bpb = better model of medical language = better summarization foundation.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The
   branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains:
   - `data/shard_00000.parquet` ... (training shards)
   - `data/shard_06542.parquet` (val shard)
   - `tokenizer/tokenizer.pkl`
   - `tokenizer/token_bytes.pt`
   If missing, tell the human to run `python prepare_medical.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Each experiment runs on a single GPU for a **fixed time budget of 5 minutes**.
Launch with: `uv run train.py`

**What you CAN do:**
- Modify `train.py` — architecture, optimizer, hyperparameters, batch size, etc.
- Try smaller models (fewer layers, smaller dim) — may generalize better on small data
- Try different sequence lengths — documents are short, so shorter context may be fine
- Adjust learning rates — small datasets often prefer lower LRs or more aggressive decay

**What you CANNOT do:**
- Modify `prepare.py` — it is read-only (fixed evaluation, dataloader, tokenizer)
- Install new packages beyond `pyproject.toml`
- Modify the `evaluate_bpb` function

**The goal: lowest val_bpb on held-out medical transcriptions.**

**VRAM**: Soft constraint. Some increase is acceptable for meaningful val_bpb gains.

**Simplicity criterion**: Same as always — simpler is better when results are equal.

**The first run**: Always establish a baseline by running train.py as-is first.

## Domain Research Ideas

When you need ideas, think about what's known for small-corpus language modeling:

1. **Smaller model size** — with ~4K documents, a tiny model may beat a large one
2. **Shorter sequence length** — clinical notes rarely exceed 512 tokens; try T=512 or T=1024
3. **More aggressive LR warmdown** — small data benefits from careful LR cooldown
4. **Higher weight decay** — regularization matters more with small datasets
5. **Larger batch size** — more gradient averaging may help with noisy small-data gradients
6. **Lower learning rate** — less aggressive updates to avoid memorizing training set
7. **Deeper vs. wider** — experiment with aspect ratio for fixed parameter count

## Output format

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract key metric: `grep "^val_bpb:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (7 chars)
2. val_bpb (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory GB, rounded to .1f (divide peak_vram_mb by 1024)
4. status: `keep`, `discard`, or `crash`
5. short description

Example:
```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline medical corpus
b2c3d4e	0.985000	20.1	keep	reduced depth=4 for small corpus
c3d4e5f	0.980000	12.3	keep	sequence_len=512 matching doc length
```

## The experiment loop

LOOP FOREVER:

1. Check git state (current branch/commit)
2. Tune `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If empty → crashed → `tail -n 50 run.log` to debug
7. Record in results.tsv
8. If val_bpb improved (lower) → keep commit (advance branch)
9. If equal or worse → `git reset --hard HEAD~1`

**Timeout**: ~5 min per experiment. Kill and discard if >10 min.

**NEVER STOP**: Once the loop begins, do NOT ask the human if you should continue.
You are autonomous. Run until manually interrupted. The human may be asleep.
