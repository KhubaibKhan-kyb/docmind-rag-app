DocMind: RAG-Powered Document Q&A

A Retrieval-Augmented Generation app that lets you upload PDF documents and ask questions about them using natural language. Built with LangChain, OpenAI, ChromaDB, and Streamlit.

What It Does

Upload one or more PDF documents through a clean web UI
Documents are split into chunks, embedded, and stored in a local vector database
Ask questions in natural language. The app retrieves the most relevant chunks and generates an answer using GPT-4.1-nano
Supports follow-up questions via conversation history.

How It Works

PDF Upload
    ↓
PyPDFLoader → RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
    ↓
OpenAIEmbeddings → ChromaDB (persisted locally)
    ↓
User Query → MMR Retriever (k=4, fetch_k=10)
    ↓
ChatPromptTemplate + Chat History → GPT-4.1-nano → Answer