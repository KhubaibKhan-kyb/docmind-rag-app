import streamlit as st
import os
import shutil
import tempfile
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind: RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --border:    #1e2330;
    --accent:    #c8f060;
    --accent2:   #60d4f0;
    --muted:     #4a5168;
    --text:      #e8ecf4;
    --text-dim:  #8892a8;
    --user-bg:   #1a2235;
    --ai-bg:     #111820;
    --radius:    14px;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.sidebar-brand {
    font-family: var(--font-head);
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--accent);
    line-height: 1.1;
    margin-bottom: 4px;
}
.sidebar-sub {
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 28px;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    background: rgba(200,240,96,0.03) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

.stButton > button {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    border: 1.5px solid var(--accent) !important;
    background: transparent !important;
    color: var(--accent) !important;
    padding: 0.5rem 1.2rem !important;
    transition: all .2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px rgba(200,240,96,0.25) !important;
}
.doc-card {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.78rem;
}
.doc-card .doc-name  { color: var(--text); margin-bottom: 4px; word-break: break-all; }
.doc-card .doc-meta  { color: var(--muted); font-size: 0.68rem; letter-spacing: 0.5px; }
.doc-card .doc-badge {
    display: inline-block;
    background: rgba(200,240,96,0.12);
    color: var(--accent);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 6px;
}
.main-header {
    font-family: var(--font-head);
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: var(--text);
    margin-bottom: 2px;
}
.main-header span { color: var(--accent); }
.main-tagline {
    color: var(--text-dim);
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 32px;
}
.chat-wrap  { display: flex; flex-direction: column; gap: 16px; margin-bottom: 20px; }
.msg-row    { display: flex; gap: 14px; align-items: flex-start; }
.msg-row.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; flex-shrink: 0;
    font-family: var(--font-head); font-weight: 700;
}
.avatar.ai  { background: rgba(96,212,240,0.15); color: var(--accent2); border: 1px solid rgba(96,212,240,0.3); }
.avatar.usr { background: rgba(200,240,96,0.12);  color: var(--accent);  border: 1px solid rgba(200,240,96,0.25); }
.bubble {
    max-width: 76%; border-radius: var(--radius);
    padding: 14px 18px; font-size: 0.85rem; line-height: 1.65;
}
.bubble.ai  { background: var(--ai-bg);   border: 1px solid var(--border); color: var(--text); }
.bubble.usr { background: var(--user-bg); border: 1px solid #243050;       color: var(--text); text-align: right; }
.msg-time { font-size: 0.62rem; color: var(--muted); margin-top: 5px; letter-spacing: 0.5px; }
.sources-label {
    font-size: 0.62rem; letter-spacing: 1.5px; text-transform: uppercase;
    color: var(--muted); margin-top: 10px; margin-bottom: 5px;
}
.source-pill {
    display: inline-block;
    background: rgba(96,212,240,0.08); border: 1px solid rgba(96,212,240,0.2);
    color: var(--accent2); border-radius: 20px; padding: 2px 10px;
    font-size: 0.65rem; margin-right: 5px; margin-bottom: 4px;
}
.status-bar {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px;
    font-size: 0.78rem; color: var(--text-dim); margin-bottom: 16px;
}
.status-bar strong { color: var(--accent); }
.empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
.empty-icon  { font-size: 3rem; margin-bottom: 16px; }
.empty-title {
    font-family: var(--font-head); font-size: 1.1rem;
    font-weight: 700; color: var(--text-dim); margin-bottom: 8px;
}
.empty-hint { font-size: 0.75rem; line-height: 1.6; }
hr { border-color: var(--border) !important; margin: 20px 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ───────────────────────────────────────────────────
for k, v in {
    "messages":     [],
    "db_ready":     False,
    "doc_meta":     [],
    "retriever":    None,
    "llm":          None,
    "prompt":       None,
    "vector_store": None,    # keep reference so we can close before deleting
    "chat_history": [],      # HumanMessage / AIMessage objects for memory
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  create_database.py  — same logic, adapted for uploaded files
# ══════════════════════════════════════════════════════════════════════════════
CHROMA_DIR = "chroma_db"

def safe_delete_chroma():
    """
    Windows fix: ChromaDB holds file locks on its .bin files.
    We must close the client reference BEFORE calling shutil.rmtree,
    otherwise Windows raises PermissionError (WinError 32).
    """
    if st.session_state.get("vector_store") is not None:
        try:
            st.session_state.vector_store._client.close()
        except Exception:
            pass
        st.session_state.vector_store = None

    if os.path.exists(CHROMA_DIR):
        for attempt in range(5):          # retry up to 5x — lock takes a moment to release
            try:
                shutil.rmtree(CHROMA_DIR)
                break
            except PermissionError:
                time.sleep(0.5)


def create_database(uploaded_files: list):
    """Mirrors create_database.py exactly."""
    all_chunks = []
    doc_meta   = []

    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        data   = PyPDFLoader(tmp_path)
        docs   = data.load()
        os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks   = splitter.split_documents(docs)

        all_chunks.extend(chunks)
        doc_meta.append({"name": f.name, "pages": len(docs), "chunks": len(chunks)})

    safe_delete_chroma()   # close + wipe old DB before rebuilding

    embedding_model = OpenAIEmbeddings()
    vs = Chroma.from_documents(all_chunks, embedding_model, persist_directory=CHROMA_DIR)

    return vs, doc_meta


# ══════════════════════════════════════════════════════════════════════════════
#  main.py  — setup + query logic, with conversation history
# ══════════════════════════════════════════════════════════════════════════════
def load_retriever_and_chain(vs):
    """Mirrors main.py setup exactly, with MessagesPlaceholder for history."""
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )

    llm = ChatOpenAI(model="gpt-4.1-nano")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant.
Use ONLY the provided context to answer the question.
If you don't know the answer, say you don't know.
Do not use any information that is not provided in the context."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         """Context:
{context}

Question:
{question}"""),
    ])

    return retriever, llm, prompt


def answer_question(query: str):
    """Mirrors main.py while-loop body, with chat history injected."""
    retriever    = st.session_state.retriever
    llm          = st.session_state.llm
    prompt       = st.session_state.prompt
    chat_history = st.session_state.chat_history

    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    sources = list({
        f"p.{doc.metadata.get('page', '?') + 1}" for doc in retrieved_docs
    })

    response = prompt.invoke({
        "context":      context,
        "question":     query,
        "chat_history": chat_history,
    })
    answer = llm.invoke(response)

    # Append this turn to memory
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer.content))

    return answer.content, sources


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-brand">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">RAG · Semantic Search</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("Process & Index"):
            prog = st.progress(0, text="Loading PDFs…")
            prog.progress(0.2, text="Splitting & embedding…")

            vs, doc_meta = create_database(uploaded_files)

            prog.progress(0.8, text="Initialising retriever…")
            retriever, llm, prompt = load_retriever_and_chain(vs)

            prog.progress(1.0, text="Ready!")
            time.sleep(0.4)
            prog.empty()

            st.session_state.db_ready     = True
            st.session_state.doc_meta     = doc_meta
            st.session_state.vector_store = vs
            st.session_state.retriever    = retriever
            st.session_state.llm          = llm
            st.session_state.prompt       = prompt
            st.session_state.messages     = []
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")

    if st.session_state.db_ready and st.session_state.doc_meta:
        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;'
            'color:#4a5168;margin-bottom:10px;">Indexed Documents</p>',
            unsafe_allow_html=True,
        )
        for d in st.session_state.doc_meta:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-name">📄 {d['name']}</div>
                <div class="doc-meta">{d['pages']} pages · {d['chunks']} chunks</div>
                <div class="doc-badge">indexed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑  Clear & Reset"):
            safe_delete_chroma()   # ← Windows-safe delete
            st.session_state.messages     = []
            st.session_state.doc_meta     = []
            st.session_state.db_ready     = False
            st.session_state.retriever    = None
            st.session_state.llm          = None
            st.session_state.prompt       = None
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        '<p style="font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;'
        'color:#4a5168;margin-bottom:10px;">Retrieval Settings</p>',
        unsafe_allow_html=True,
    )
    k_docs = st.slider("Chunks retrieved (k)", 2, 8, 4)
    if st.session_state.retriever:
        st.session_state.retriever.search_kwargs["k"] = k_docs


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
_, col_main, _ = st.columns([0.5, 9, 0.5])

with col_main:
    st.markdown('<div class="main-header">Ask your <span>documents.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-tagline">Semantic retrieval · GPT-4.1-nano · ChromaDB</div>', unsafe_allow_html=True)

    if st.session_state.db_ready:
        total_chunks = sum(d["chunks"] for d in st.session_state.doc_meta)
        total_docs   = len(st.session_state.doc_meta)
        st.markdown(
            f'<div class="status-bar">Index ready — '
            f'<strong>{total_docs}</strong> document{"s" if total_docs > 1 else ""} · '
            f'<strong>{total_chunks}</strong> chunks indexed</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-bar">Upload PDFs in the sidebar and click '
            '<strong>Process &amp; Index</strong> to begin.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.messages:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="avatar usr">U</div>
                    <div>
                        <div class="bubble usr">{msg["content"]}</div>
                        <div class="msg-time" style="text-align:right">{msg.get("time","")}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                sources_html = ""
                if msg.get("sources"):
                    pills = "".join(
                        f'<span class="source-pill">{s}</span>' for s in msg["sources"]
                    )
                    sources_html = f'<div class="sources-label">Sources</div>{pills}'
                st.markdown(f"""
                <div class="msg-row">
                    <div class="avatar ai">AI</div>
                    <div>
                        <div class="bubble ai">{msg["content"]}{"<br/>" + sources_html if sources_html else ""}</div>
                        <div class="msg-time">{msg.get("time","")}</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <div class="empty-title">No conversation yet</div>
            <div class="empty-hint">
                Upload your documents, index them,<br>
                then ask anything about their content.
            </div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.db_ready:
        query = st.chat_input("Ask something about your documents…")
        if query:
            ts = time.strftime("%H:%M")
            st.session_state.messages.append({"role": "user", "content": query, "time": ts})
            with st.spinner("Retrieving & generating…"):
                try:
                    answer, sources = answer_question(query)
                except Exception as e:
                    answer, sources = f"Error: {e}", []
            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "time": time.strftime("%H:%M"),
            })
            st.rerun()
    else:
        st.chat_input("Index your documents first…", disabled=True)
