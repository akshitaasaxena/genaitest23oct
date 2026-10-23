"""
server.py — FastAPI wrapper around the existing RAG pipeline
Wraps the same retriever + LCEL chain used in ask.py behind a single
POST /ask endpoint, so a frontend (index.html) can call it over HTTP.

Embedding model : sentence-transformers/all-MiniLM-L6-v2 (local, no quota)
LLM             : gemini-3.8-flash via Google Generative AI API
                  (gemini-2.5-flash was deprecated for this API key — 404'd
                  with "no longer available to new users")

No retrieval, prompt, or embedding logic is rewritten here — this file only
wraps the existing chain from ask.py in an HTTP interface.
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── 1. Load environment variables ────────────────────────────────────────────
# GOOGLE_API_KEY is needed for Gemini (answer generation only).
# Embedding is local — no key needed for that step.
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")

# ── 2. Load the local embedding model ────────────────────────────────────────
# CRITICAL: Must be identical to the model used in ingest.py, or retrieval
# silently returns wrong/garbage chunks (query and document vectors would
# live in different embedding spaces).
print("🤖 Loading local embedding model ...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ── 3. Load the existing Chroma vector database ──────────────────────────────
PERSIST_DIR = "db"

if not os.path.exists(PERSIST_DIR):
    raise FileNotFoundError(
        f"No database found at ./{PERSIST_DIR}. Run ingest.py first."
    )

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings,
)
print(f"📂 Loaded Chroma DB from ./{PERSIST_DIR} "
      f"({vectorstore._collection.count()} chunks)\n")

# ── 4. Create a retriever ────────────────────────────────────────────────────
# k=3 returns the 3 most semantically similar chunks for each query.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── 5. Build the LCEL chain (identical to ask.py) ────────────────────────────

# 5a. Format retrieved Document objects into a single context string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5b. System prompt — strictly constrains the model to catalogue context only.
SYSTEM_PROMPT = (
    "Answer the question using ONLY the context below, which is from the "
    "Orient BlackSwan History catalogue. Do not use outside knowledge. "
    "If the answer is not in the context, or if you are even slightly unsure, "
    "reply with exactly the two words: Not found — nothing else."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# 5c. LLM — Gemini 3.8 Flash (gemini-2.5-flash 404'd as deprecated for this key).
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# 5d. Assemble the LCEL chain:
#   question ──► retriever ──► format_docs ──► prompt ──► llm ──► string
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ── 6. Reusable answer function ──────────────────────────────────────────────
def get_answer(question: str) -> dict:
    """
    Runs a question through the existing RAG chain and returns the answer
    plus cited source page numbers, in the shape expected by index.html:
        {"answer": "...", "sources": ["page 12", "page 14"]}
    Sources are left empty when the chain replies "Not found", since there's
    nothing to cite.
    """
    # Retrieve the relevant chunks directly so we can inspect their metadata
    # for source citation, separate from running the full answer chain.
    source_docs = retriever.invoke(question)

    # Run the full LCEL chain to get the answer from Gemini.
    answer = chain.invoke(question).strip()

    sources = []
    if answer != "Not found":
        pages = []
        for doc in source_docs:
            page_num = doc.metadata.get("page")
            if page_num is not None:
                # PyPDFLoader uses 0-based page index → +1 for human-readable.
                pages.append(f"page {int(page_num) + 1}")
        # Deduplicate while preserving retrieval order.
        seen = set()
        sources = [p for p in pages if not (p in seen or seen.add(p))]

    return {"answer": answer, "sources": sources}


# ── 7. FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(title="Orient BlackSwan History Catalogue — RAG API")

# Enable CORS so a static index.html opened in the browser (or served from
# localhost) can call this API without being blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5500",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",
        "null",  # browsers send Origin: null for file:// pages
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
async def ask(request: Request):
    """
    Expects JSON: {"question": "..."}
    Returns JSON: {"answer": "...", "sources": [...]}
    Returns 400 if "question" is missing or empty.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Request body must be valid JSON."},
        )

    question = body.get("question", "").strip() if isinstance(body, dict) else ""

    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing required field: 'question'."},
        )

    result = get_answer(question)
    return JSONResponse(content=result)


# ── 8. Run with: uvicorn server:app --reload ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
