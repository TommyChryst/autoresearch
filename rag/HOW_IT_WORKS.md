# How the RAG Chatbot Works — Full Flow

## The Two Separate Things We Built

This project has two independent pieces that work together:

1. **The trained LLM** — a small transformer trained from scratch on MTSamples via Karpathy's autoresearch loop (`train.py`)
2. **The RAG chatbot** — a retrieval-augmented chat interface using Ollama + ChromaDB (`rag/`)

They share the same **dataset** (MTSamples), but the RAG chatbot does *not* use the model we trained. It uses a pre-built model served by Ollama. The trained model is a separate research artifact.

---

## What Ollama Is

Ollama is a local inference server. It:
- Downloads and stores pre-trained models (e.g. `phi3.5`, `qwen2.5:3b`)
- Exposes a REST API at `http://localhost:11434`
- Runs inference on your machine — no internet, no API key, fully private

When you run `ollama pull phi3.5`, it downloads Microsoft's Phi-3.5 model weights locally. When you call `ollama.chat(...)`, it sends tokens through that model and streams back a response — like a local version of the ChatGPT API.

---

## The RAG Flow, Step by Step

```
User types a prompt
        │
        ▼
┌─────────────────────┐
│   embed_mtsamples   │  (run once at setup)
│   .py               │
│                     │
│  mtsamples.csv      │
│  ~4,000 clinical    │──► chunk into ~400-word pieces
│  transcriptions     │──► embed with all-MiniLM-L6-v2
│                     │──► store in ChromaDB (local vector store)
└─────────────────────┘
        │
        │  chroma_db/ now on disk
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   chat.py / app.py                  │
│                                                     │
│  1. RETRIEVE                                        │
│     embed the user's query with the same model      │
│     find top-5 most similar chunks in ChromaDB      │
│     (cosine similarity over ~15K vectors)           │
│                                                     │
│  2. BUILD PROMPT                                    │
│     SYSTEM_PROMPT (SOAP rules + format)             │
│     + retrieved MTSamples examples (as context)     │
│     + conversation history                          │
│     + user's question                               │
│                                                     │
│  3. CALL OLLAMA                                     │
│     send the full prompt to ollama.chat()           │
│     Ollama runs phi3.5 locally                      │
│     stream tokens back to terminal / Streamlit UI   │
└─────────────────────────────────────────────────────┘
        │
        ▼
SOAP note printed to user
```

---

## Why RAG Instead of Just Prompting?

Without RAG, the model has no knowledge of MTSamples — it only knows what it was trained on (generic internet text). It would generate plausible but generic SOAP notes with no grounding in real clinical transcriptions.

With RAG:
- The retrieved chunks give the model **real examples** of how clinical notes are written
- The model uses those examples for **phrasing, structure, and clinical style**
- It's not fine-tuned — it's just given good context at inference time

This is much faster to build than fine-tuning and works well for style/format transfer.

---

## How the Three RAG Files Connect

| File | What it does | Run when |
|------|-------------|----------|
| `embed_mtsamples.py` | Reads CSV → chunks → embeds → writes `chroma_db/` | Once at setup |
| `chat.py` | Terminal chat loop using Ollama | `python rag/chat.py` |
| `app.py` | Streamlit web UI (same logic as chat.py) | `streamlit run rag/app.py` |

`chat.py` and `app.py` are the same pipeline, different UIs. Both use the same `SYSTEM_PROMPT`, same Chroma retrieval, same Ollama call.

---

## How It Relates to the Trained Model

```
mtsamples.csv
     │
     ├──► data_prep.py ──► medical_train.txt ──► prepare_medical.py
     │                                                │
     │                                          parquet shards
     │                                          + tokenizer
     │                                                │
     │                                           train.py
     │                                          (5-min loop)
     │                                          val_bpb metric
     │                                          (autoresearch)
     │
     └──► embed_mtsamples.py ──► chroma_db/
                                      │
                                 chat.py / app.py
                                      │
                                  Ollama (phi3.5)
                                      │
                                 SOAP notes to user
```

The trained model learns the *statistical patterns* of medical language (useful for evaluating how well it has compressed the corpus). The RAG chatbot uses MTSamples as a *retrieval index* to ground a general-purpose model at inference time. Same data, two completely different uses.

---

## Quick Setup Recap

```bash
# 1. Install Ollama and pull a model
ollama pull phi3.5

# 2. Build the vector index (one time)
python rag/embed_mtsamples.py

# 3. Run the chatbot
python rag/chat.py           # terminal
streamlit run rag/app.py     # web UI
```
