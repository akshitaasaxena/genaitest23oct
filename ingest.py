"""
ingest.py — RAG Ingestion Pipeline
Loads a PDF, chunks it, embeds locally with HuggingFace sentence-transformers,
and stores in a local persistent Chroma vector database.

Embedding model: sentence-transformers/all-MiniLM-L6-v2
  ✅ Runs 100% locally — no API calls, no quota, no API key needed for embedding.
  ✅ Must match the embedding model used in ask.py exactly.
"""

import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── 1. Load environment variables ────────────────────────────────────────────
# GOOGLE_API_KEY is not needed for embedding (local model), but keep .env
# loading here so the file stays consistent with ask.py which needs it for
# the Gemini chat model.
load_dotenv()

# ── 2. Recreate the "db" folder from scratch ─────────────────────────────────
# The old ./db was built with GoogleGenerativeAIEmbeddings (gemini-embedding-001).
# HuggingFace vectors live in a different embedding space — mixing models in one
# Chroma collection silently breaks retrieval. Wipe and rebuild cleanly.
PERSIST_DIR = "db"

if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"🗑️  Deleted old ./{PERSIST_DIR} (incompatible embedding model).")

# ── 3. Load the PDF ──────────────────────────────────────────────────────────
# PyPDFLoader reads each page as a separate Document object.
# Each Document automatically carries metadata: page number and source filename.
PDF_PATH = "History_2024-25.pdf"

print(f"📄 Loading PDF: {PDF_PATH}")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

total_pages = len(pages)
print(f"   Loaded {total_pages} pages.")

# ── 4. Split pages into chunks ───────────────────────────────────────────────
# RecursiveCharacterTextSplitter tries to split on natural boundaries
# (paragraphs → sentences → words) before falling back to raw character cuts.
#   • chunk_size=500  — each chunk is at most 500 characters
#   • chunk_overlap=50 — neighbouring chunks share 50 characters of context,
#     so no sentence is lost at a boundary
# Metadata (page number, source) propagates automatically from parent pages.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = text_splitter.split_documents(pages)

total_chunks = len(chunks)
print(f"✂️  Split into {total_chunks} chunks (500 chars, 50 overlap).")

# ── 5. Create local embeddings ───────────────────────────────────────────────
# HuggingFaceEmbeddings downloads the model once to ~/.cache/huggingface on
# first run, then uses it entirely offline. No API key or quota required.
#
# Model: all-MiniLM-L6-v2
#   • 384-dimensional vectors, fast inference, strong semantic quality.
#   • ~90 MB download (one-time). Subsequent runs use the local cache.
#
# ⚠️  IMPORTANT: ask.py must use the exact same model_name or retrieval breaks —
#     query and document vectors must live in the same embedding space.
print("🤖 Loading local embedding model (all-MiniLM-L6-v2) ...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ── 6. Store all chunks in Chroma ────────────────────────────────────────────
# Chroma stores vectors locally on disk in the "db" folder.
# from_documents embeds every chunk and persists to disk in one call —
# no batching or rate limiting needed since it's all local.
print(f"💾 Embedding {total_chunks} chunks and storing in ./{PERSIST_DIR} ...")
print(f"   (Running locally — no API quota used)")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)

# ── 7. Done ──────────────────────────────────────────────────────────────────
print()
print(f"📊 Summary:")
print(f"   • Pages loaded:   {total_pages}")
print(f"   • Chunks created: {total_chunks}")
print(f"✅ Stored {total_chunks} chunks in ./{PERSIST_DIR}")
