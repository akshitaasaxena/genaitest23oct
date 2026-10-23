"""
ask.py — RAG Query Interface
Loads the Chroma vector DB built by ingest.py, retrieves relevant chunks,
and answers questions using Gemini 3.8 Flash via a LangChain LCEL chain.

Embedding model : sentence-transformers/all-MiniLM-L6-v2 (local, no quota)
LLM             : gemini-3.8-flash via Google Generative AI API
                  (gemini-2.5-flash was deprecated for this API key — 404'd
                  with "no longer available to new users")
"""

import os
from dotenv import load_dotenv
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
# CRITICAL: Must be identical to the model used in ingest.py.
# Mismatched models → query and document vectors in different spaces →
# retrieval silently returns wrong/garbage chunks.
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

# ── 5. Build the LCEL chain ──────────────────────────────────────────────────

# 5a. Format retrieved Document objects into a single context string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5b. System prompt — strictly constrains the model to catalogue context only.
#     "even slightly unsure" + "nothing else" closes the hallucination loophole
#     so Q3 (Physics price) and Q4 (Minister of Education) reliably → Not found.
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
#     temperature=0 → maximally deterministic, discourages hallucination.
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

# ── 6. Interactive question-answer loop ──────────────────────────────────────
print("📚 Orient BlackSwan History Catalogue — RAG Assistant")
print("   Type your question and press Enter. Type 'exit' to quit.\n")

while True:
    question = input("Question: ").strip()

    if not question:
        continue

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve the relevant chunks directly so we can inspect their metadata
    # for source citation, separate from running the full answer chain.
    source_docs = retriever.invoke(question)

    # Run the full LCEL chain to get the answer from Gemini.
    answer = chain.invoke(question)

    print(f"\nAnswer: {answer}")

    # ── 7. Print source page numbers ────────────────────────────────────────
    # Only cite sources when the model found an answer in the context.
    # Page numbers come from metadata attached by PyPDFLoader in ingest.py.
    # PyPDFLoader uses 0-based page index → add 1 for human-readable numbers.
    if answer.strip() != "Not found":
        pages = []
        for doc in source_docs:
            page_num = doc.metadata.get("page")
            if page_num is not None:
                pages.append(f"page {int(page_num) + 1}")
        if pages:
            # Deduplicate while preserving retrieval order.
            seen = set()
            unique_pages = [p for p in pages if not (p in seen or seen.add(p))]
            print(f"Source: {', '.join(unique_pages)}")

    print()  # blank line between Q&A pairs
