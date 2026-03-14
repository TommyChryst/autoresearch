"""
app.py — Streamlit web UI for the MTSamples RAG chatbot.

Usage:
    streamlit run rag/app.py
    streamlit run rag/app.py -- --model qwen2.5:3b

Requires:
    pip install streamlit chromadb sentence-transformers ollama
    Ollama running with a model pulled (e.g. ollama pull phi3.5)
    Chroma DB built (python rag/embed_mtsamples.py)
"""

import os
import sys
import argparse

import streamlit as st

# ---------------------------------------------------------------------------
# Paths / defaults (must match chat.py)
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "medical-soap"
DEFAULT_DB = os.path.join(os.path.dirname(__file__), "chroma_db")
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "mtsamples"
MAX_HISTORY_TURNS = 6
MAX_CHUNK_CHARS = 800

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
# Resource loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Chroma DB...")
def load_collection(db_path: str, embed_model: str):
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError:
        st.error("Run: pip install chromadb sentence-transformers")
        sys.exit(1)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


@st.cache_data(show_spinner=False)
def list_specialties(db_path: str, embed_model: str) -> list[str]:
    """Return sorted unique specialty values from the collection metadata."""
    col = load_collection(db_path, embed_model)
    # Sample up to 5000 items to get specialty list
    results = col.get(limit=5000, include=["metadatas"])
    specialties = set()
    for meta in results["metadatas"]:
        s = meta.get("specialty", "").strip()
        if s:
            specialties.add(s)
    return sorted(specialties)


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------
def retrieve(collection, query: str, top_k: int, specialty: str | None) -> list[dict]:
    where = {"specialty": {"$eq": specialty}} if specialty else None
    kwargs = dict(query_texts=[query], n_results=top_k)
    if where:
        kwargs["where"] = where
    try:
        results = collection.query(**kwargs)
    except Exception:
        results = collection.query(query_texts=[query], n_results=top_k)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "meta": meta, "distance": dist})
    return chunks


def build_messages(user_query: str, chunks: list[dict], history: list[dict]) -> list[dict]:
    examples_text = ""
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        header = f"[Example {i} — {meta.get('specialty', 'Unknown')} | {meta.get('sample_name', '')}]"
        body = chunk["text"][:MAX_CHUNK_CHARS]
        if len(chunk["text"]) > MAX_CHUNK_CHARS:
            body += "..."
        examples_text += f"\n{header}\n{body}\n"

    system = SYSTEM_PROMPT
    if examples_text:
        system += f"\n\n---\nRELEVANT MTSamples EXAMPLES:\n{examples_text}---"

    messages = [{"role": "system", "content": system}]
    messages.extend(history[-(MAX_HISTORY_TURNS * 2):])
    messages.append({"role": "user", "content": user_query})
    return messages


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🏥", layout="wide")
    st.title("🏥 Medical RAG Chatbot")
    st.caption("Powered by MTSamples + Ollama — fully local, fully private")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Settings")

        model = st.text_input("Ollama model", value=DEFAULT_MODEL)
        embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
        db_path = st.text_input("Chroma DB path", value=DEFAULT_DB)
        top_k = st.slider("Retrieved chunks (top-K)", min_value=1, max_value=15, value=5)

        st.divider()

        # Specialty filter
        try:
            specialties = list_specialties(db_path, embed_model)
            specialty_options = ["(All specialties)"] + specialties
            selected_specialty = st.selectbox("Specialty filter", specialty_options)
            specialty = None if selected_specialty == "(All specialties)" else selected_specialty
        except Exception as e:
            st.warning(f"Could not load specialties: {e}")
            specialty = None

        st.divider()
        show_sources = st.checkbox("Show retrieved sources", value=True)

        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.rerun()

        st.divider()
        st.markdown("**Quick prompts:**")
        quick_prompts = [
            "Generate a SOAP note for a 55yo with chest pain and hypertension on lisinopril",
            "What are the most common diagnoses in orthopedic notes?",
            "Summarize the typical structure of a cardiology consultation note",
            "What medications appear most frequently in the dataset?",
        ]
        for prompt in quick_prompts:
            if st.button(prompt[:60] + "...", key=prompt):
                st.session_state.pending_prompt = prompt

    # ---- Init state ----
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []

    # ---- Load collection ----
    try:
        collection = load_collection(db_path, embed_model)
        chunk_count = collection.count()
        st.sidebar.success(f"DB loaded: {chunk_count:,} chunks")
    except Exception as e:
        st.error(f"Failed to load Chroma DB: {e}\n\nRun: `python rag/embed_mtsamples.py`")
        return

    # ---- Display chat history ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and show_sources:
                with st.expander("Sources retrieved"):
                    for i, src in enumerate(msg["sources"], 1):
                        meta = src["meta"]
                        st.markdown(
                            f"**{i}.** {meta.get('specialty', '?')} | "
                            f"{meta.get('sample_name', '?')} *(dist={src['distance']:.3f})*"
                        )
                        st.caption(src["text"][:300] + "...")

    # ---- Handle pending quick prompt ----
    pending = st.session_state.pop("pending_prompt", None)

    # ---- Chat input ----
    user_input = st.chat_input("Ask about medical notes, generate a SOAP note, ...") or pending

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve
        with st.spinner("Retrieving relevant examples..."):
            chunks = retrieve(collection, user_input, top_k, specialty)

        # Build messages
        messages = build_messages(user_input, chunks, st.session_state.history)

        # Stream response
        with st.chat_message("assistant"):
            try:
                import ollama as _ollama

                stream = _ollama.chat(model=model, messages=messages, stream=True)
                response_text = st.write_stream(
                    (chunk["message"]["content"] for chunk in stream if chunk["message"]["content"])
                )
            except Exception as e:
                st.error(f"Ollama error: {e}\n\nIs Ollama running? Is model '{model}' pulled?")
                response_text = ""

            if response_text and show_sources:
                with st.expander("Sources retrieved"):
                    for i, src in enumerate(chunks, 1):
                        meta = src["meta"]
                        st.markdown(
                            f"**{i}.** {meta.get('specialty', '?')} | "
                            f"{meta.get('sample_name', '?')} *(dist={src['distance']:.3f})*"
                        )
                        st.caption(src["text"][:300] + "...")

        # Update state
        if response_text:
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": chunks,
            })
            st.session_state.history.append({"role": "user", "content": user_input})
            st.session_state.history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
