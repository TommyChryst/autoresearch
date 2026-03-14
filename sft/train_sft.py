"""
train_sft.py — QLoRA fine-tune Phi-3.5-mini-instruct on synthetic SOAP note pairs.

Usage:
    python sft/train_sft.py                      # defaults: 3 epochs, lr=2e-4, r=16
    python sft/train_sft.py --epochs 5 --lr 1e-4
    python sft/train_sft.py --no-4bit            # fp16 mode (needs more VRAM, for debugging)

Input:  sft/pairs.jsonl — {"prompt": "...", "response": "..."}
Output: sft/output/     — LoRA adapter weights

Expected training time on T4 (16GB): ~15 min for 500 pairs × 3 epochs
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PAIRS_FILE = SCRIPT_DIR / "pairs.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "output"

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

# LoRA config — r=16 for low-data regime (500 pairs)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training defaults
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-4
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 8       # effective batch = 16
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_WARMUP_STEPS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Phi-3.5-mini on SOAP pairs")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--pairs", type=Path, default=PAIRS_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    args = parser.parse_args()

    if not args.pairs.exists():
        raise FileNotFoundError(
            f"pairs.jsonl not found at {args.pairs}\n"
            "Run: python sft/generate_pairs.py --count 500"
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Verify the <|assistant|> special token ID
    asst_id = tokenizer.encode("<|assistant|>", add_special_tokens=False)[0]
    print(f"<|assistant|> token ID: {asst_id}")

    # ---- Pre-tokenize and set labels BEFORE the trainer sees the data ----
    # Labels = input_ids with all tokens up to (and including) <|assistant|> masked to -100.
    # Using standard Trainer (not SFTTrainer) so no trl label overrides happen.
    pairs = load_pairs(args.pairs)

    processed = []
    skipped = 0
    for p in pairs:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": p["response"]},
            ],
            tokenize=False,
        )
        enc = tokenizer(
            text,
            truncation=True,
            max_length=DEFAULT_MAX_SEQ_LEN,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"]
        labels = list(input_ids)

        # Find last <|assistant|> token; mask everything up to and including it
        response_start = None
        for j in range(len(input_ids) - 1, -1, -1):
            if input_ids[j] == asst_id:
                response_start = j + 1
                break

        if response_start is not None:
            for k in range(response_start):
                labels[k] = -100
        else:
            skipped += 1
            continue

        processed.append({
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        })

    print(f"Pre-tokenized: {len(processed)} examples ({skipped} skipped)")

    s = processed[0]
    n_resp = sum(1 for l in s["labels"] if l != -100)
    n_total = len(s["input_ids"])
    print(f"Sample[0]: {n_total} tokens total, {n_resp} response tokens (loss target)")
    assert n_resp > 0, "No response tokens found! Check <|assistant|> token ID."

    dataset = Dataset.from_list(processed)

    # ---- Quantization config ----
    if not args.no_4bit:
        print("Using 4-bit quantization (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("Using fp16 (no quantization)")
        bnb_config = None

    # ---- Load base model ----
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not args.no_4bit else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Collator: pads input_ids AND labels, uses -100 for padded label positions ----
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ---- Training config ----
    # Use standard Trainer (NOT SFTTrainer) to guarantee labels are not overridden.
    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=DEFAULT_BATCH_SIZE,
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=DEFAULT_WARMUP_STEPS,
        bf16=not args.no_4bit,
        fp16=args.no_4bit,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,      # recompute activations to save ~40% VRAM
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # ---- Train ----
    print(f"\nStarting training: {args.epochs} epochs, lr={args.lr}, r={args.lora_r}")
    print(f"Effective batch size: {DEFAULT_BATCH_SIZE * DEFAULT_GRAD_ACCUM}")
    print(f"Output: {args.output}\n")

    trainer.train()

    # ---- Save adapter ----
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nLoRA adapter saved to {args.output}")


if __name__ == "__main__":
    main()
