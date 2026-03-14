"""
push_to_ollama.py — Merge LoRA adapter into Phi-3.5, convert to GGUF, register with Ollama.

This script automates the full pipeline:
  1. Load Phi-3.5-mini-instruct base + LoRA adapter → merge → save HF format
  2. Convert merged model to GGUF via llama.cpp's convert_hf_to_gguf.py
  3. Quantize to Q4_K_M via llama-quantize
  4. Write Ollama Modelfile with SYSTEM_PROMPT from rag/app.py
  5. Register: ollama create medical-soap -f Modelfile

Prerequisites (on RunPod):
    apt-get install -y cmake build-essential
    git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
    cd /opt/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)
    pip install -r /opt/llama.cpp/requirements.txt

After running:
    Change DEFAULT_MODEL = "medical-soap" in rag/app.py and rag/chat.py
    Then: streamlit run rag/app.py

Usage:
    python sft/push_to_ollama.py
    python sft/push_to_ollama.py --adapter sft/output --model-name medical-soap
    python sft/push_to_ollama.py --skip-convert  # if GGUF already exists
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
RAG_APP = REPO_ROOT / "rag" / "app.py"

LLAMA_CPP_DIR = Path("/opt/llama.cpp")
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
LLAMA_QUANTIZE = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
DEFAULT_ADAPTER = SCRIPT_DIR / "output"
DEFAULT_MERGED = SCRIPT_DIR / "merged"
DEFAULT_GGUF = SCRIPT_DIR / "medical-soap.gguf"
DEFAULT_GGUF_Q4 = SCRIPT_DIR / "medical-soap-q4km.gguf"
DEFAULT_MODELFILE = SCRIPT_DIR / "Modelfile"
DEFAULT_MODEL_NAME = "medical-soap"


# ---------------------------------------------------------------------------
# Extract SYSTEM_PROMPT from rag/app.py
# ---------------------------------------------------------------------------
def extract_system_prompt() -> str:
    """Parse SYSTEM_PROMPT constant from rag/app.py."""
    if not RAG_APP.exists():
        raise FileNotFoundError(f"rag/app.py not found at {RAG_APP}")

    source = RAG_APP.read_text(encoding="utf-8")
    # Find the triple-quoted string assigned to SYSTEM_PROMPT
    marker = 'SYSTEM_PROMPT = """\\'
    start = source.find(marker)
    if start == -1:
        raise ValueError("Could not find SYSTEM_PROMPT in rag/app.py")

    start = source.find('"""', start) + 3
    end = source.find('"""', start)
    prompt = source[start:end].strip()
    # Remove trailing backslash if present (line continuation)
    prompt = prompt.rstrip("\\").strip()
    return prompt


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def step_merge(adapter_path: Path, merged_path: Path):
    print(f"\n[1/5] Merging LoRA adapter into base model...")
    print(f"  Base: {BASE_MODEL}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {merged_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading base model (fp16)...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("  Merging and unloading...")
    model = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model to {merged_path}...")
    model.save_pretrained(str(merged_path), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_path))
    print("  Merge complete.")


def step_convert(merged_path: Path, gguf_path: Path):
    print(f"\n[2/5] Converting to GGUF...")
    if not CONVERT_SCRIPT.exists():
        print(f"ERROR: {CONVERT_SCRIPT} not found.")
        print("Install llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp")
        print("  cd /opt/llama.cpp && cmake -B build && cmake --build build -j$(nproc)")
        sys.exit(1)

    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(merged_path),
        "--outfile", str(gguf_path),
        "--outtype", "f16",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  GGUF saved: {gguf_path}")


def step_quantize(gguf_path: Path, q4_path: Path):
    print(f"\n[3/5] Quantizing to Q4_K_M...")
    if not LLAMA_QUANTIZE.exists():
        print(f"ERROR: {LLAMA_QUANTIZE} not found. Build llama.cpp first.")
        sys.exit(1)

    cmd = [str(LLAMA_QUANTIZE), str(gguf_path), str(q4_path), "Q4_K_M"]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Quantized model: {q4_path}")
    size_gb = q4_path.stat().st_size / 1e9
    print(f"  File size: {size_gb:.2f} GB")


def step_modelfile(q4_path: Path, modelfile_path: Path, system_prompt: str):
    print(f"\n[4/5] Writing Modelfile...")
    # Escape any double quotes in system prompt
    safe_prompt = system_prompt.replace('"', '\\"')
    content = f"""FROM {q4_path.resolve()}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_predict 800

SYSTEM \"{safe_prompt}\"
"""
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"  Modelfile written: {modelfile_path}")


def step_ollama_create(modelfile_path: Path, model_name: str):
    print(f"\n[5/5] Registering with Ollama as '{model_name}'...")
    if shutil.which("ollama") is None:
        print("ERROR: ollama not found in PATH. Install from https://ollama.ai")
        sys.exit(1)

    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"\nSuccess! Model registered as: {model_name}")
    print(f"\nNext steps:")
    print(f"  1. Test: ollama run {model_name} 'Generate a SOAP note for a 45yo female with DM2 and foot ulcer'")
    print(f"  2. Update rag/app.py:  DEFAULT_MODEL = \"{model_name}\"")
    print(f"  3. Update rag/chat.py: DEFAULT_MODEL = \"{model_name}\"")
    print(f"  4. Restart RAG chatbot: streamlit run rag/app.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge LoRA → GGUF → Ollama")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--merged", type=Path, default=DEFAULT_MERGED)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--gguf-q4", type=Path, default=DEFAULT_GGUF_Q4)
    parser.add_argument("--modelfile", type=Path, default=DEFAULT_MODELFILE)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge (merged model already exists)")
    parser.add_argument("--skip-convert", action="store_true", help="Skip GGUF conversion (GGUF already exists)")
    parser.add_argument("--skip-quantize", action="store_true", help="Skip quantization")
    args = parser.parse_args()

    print("=== push_to_ollama.py ===")
    print(f"Adapter:    {args.adapter}")
    print(f"Model name: {args.model_name}")

    # Extract SYSTEM_PROMPT
    try:
        system_prompt = extract_system_prompt()
        print(f"SYSTEM_PROMPT: {len(system_prompt)} chars extracted from rag/app.py")
    except Exception as e:
        print(f"WARNING: Could not extract SYSTEM_PROMPT: {e}")
        system_prompt = "You are an expert clinical documentation specialist. Write precise SOAP notes."

    if not args.skip_merge:
        step_merge(args.adapter, args.merged)

    if not args.skip_convert:
        step_convert(args.merged, args.gguf)

    if not args.skip_quantize:
        step_quantize(args.gguf, args.gguf_q4)
        final_gguf = args.gguf_q4
    else:
        final_gguf = args.gguf

    step_modelfile(final_gguf, args.modelfile, system_prompt)
    step_ollama_create(args.modelfile, args.model_name)


if __name__ == "__main__":
    main()
