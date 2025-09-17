from __future__ import annotations  # must be first

import os, io, logging
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter

import psycopg2  # pip install psycopg2-binary
from pypdf import PdfReader     # pip install pypdf

# LangSmith observability
from langsmith.run_helpers import traceable, trace  # langsmith==0.4.28

# -----------------------------
# Env
# -----------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "llamadb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

PGVECTOR_SCHEMA = os.getenv("PGVECTOR_SCHEMA", "public")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "data_llama_index")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Chunking defaults (tweak as you like)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))          # ~700 chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))    # ~100 char overlap

logger = logging.getLogger("uvicorn.error")

# -----------------------------
# Embeddings (OpenAI -> HF fallback) + derive dim
# -----------------------------
def init_embeddings() -> dict:
    try:
        if OPENAI_API_KEY:
            from llama_index.embeddings.openai import OpenAIEmbedding
            model_name = "text-embedding-3-small"  # 1536 dims
            Settings.embed_model = OpenAIEmbedding(model=model_name, api_key=OPENAI_API_KEY)
        else:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
            Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        dim = len(Settings.embed_model.get_text_embedding("probe"))
        return {
            "provider": "openai" if OPENAI_API_KEY else "huggingface",
            "model": model_name,
            "embed_dim": dim,
        }
    except Exception:
        logger.exception("Embedding init failed")
        raise

def init_llm() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    return Groq(api_key=GROQ_API_KEY, model="llama3-8b-8192")

# -----------------------------
# Ensure pgvector extension + table exist
# -----------------------------
def ensure_pgvector_ready(embed_dim: int):
    conn = psycopg2.connect(
        host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
        user=POSTGRES_USER, password=POSTGRES_PASSWORD
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{PGVECTOR_SCHEMA}";')
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS "{PGVECTOR_SCHEMA}"."{PGVECTOR_TABLE}" (
                  id BIGSERIAL PRIMARY KEY,
                  node_id TEXT,
                  text TEXT,
                  metadata_ JSONB,
                  embedding VECTOR({embed_dim})
                );
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS "idx_{PGVECTOR_TABLE}_embedding"
                ON "{PGVECTOR_SCHEMA}"."{PGVECTOR_TABLE}"
                USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS "idx_{PGVECTOR_TABLE}_node_id"
                ON "{PGVECTOR_SCHEMA}"."{PGVECTOR_TABLE}" (node_id);
            """)
    finally:
        conn.close()

# -----------------------------
# Build store + index
# -----------------------------
def init_vector_index(embed_dim: int, llm: Groq) -> VectorStoreIndex:
    ensure_pgvector_ready(embed_dim)
    vector_store = PGVectorStore.from_params(
        database=POSTGRES_DB,
        host=POSTGRES_HOST,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        schema_name=PGVECTOR_SCHEMA,
        table_name=PGVECTOR_TABLE,
        embed_dim=embed_dim,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, llm=llm)

# -----------------------------
# Utilities: PDF -> Documents -> Chunks
# -----------------------------
def extract_pdf_texts(contents: bytes) -> List[str]:
    """Return a list of per-page strings (may include empty strings)."""
    reader = PdfReader(io.BytesIO(contents))
    pages: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return pages

def build_page_documents(pages: List[str], filename: str) -> List[Document]:
    """Create one Document per page (with metadata)."""
    docs: List[Document] = []
    for i, t in enumerate(pages, start=1):
        cleaned = t.strip()
        if not cleaned:
            # Keep empty pages out to avoid empty chunks
            continue
        docs.append(Document(text=cleaned, metadata={"filename": filename, "page_number": i}))
    return docs

def chunk_documents(docs: List[Document]) -> list:
    """Chunk documents into overlapping passages for better retrieval."""
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    return nodes

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Llama API")
INDEX: Optional[VectorStoreIndex] = None
STARTUP_INFO = {}

@traceable
@app.on_event("startup")
def on_startup():
    global INDEX, STARTUP_INFO
    with trace("startup", run_type="chain", tags=["boot"]) as run:
        emb = init_embeddings()
        llm = init_llm()
        INDEX = init_vector_index(emb["embed_dim"], llm)
        STARTUP_INFO = {
            "embeddings": emb,
            "llm": {"provider": "groq", "model": llm.model},
            "pgvector": {"schema": PGVECTOR_SCHEMA, "table": PGVECTOR_TABLE},
            "chunking": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        }
        try:
            run.add_outputs(STARTUP_INFO)
        except Exception:
            pass

class IngestRequest(BaseModel):
    text: str
    id: Optional[str] = None

class QueryRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    if INDEX is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    return {"ok": True, **STARTUP_INFO}

# -------- Plain-text ingest --------
@traceable
@app.post("/ingest")
def ingest_doc(req: IngestRequest):
    with trace("ingest_doc", run_type="chain",
               inputs={"id": req.id, "text_len": len(req.text or "")},
               tags=["ingest", "text"]) as run:
        try:
            if not req.text.strip():
                raise HTTPException(status_code=422, detail="text is required")
            doc = Document(text=req.text, doc_id=req.id) if req.id else Document(text=req.text)
            with trace("index.insert", run_type="retriever", inputs={"has_id": bool(req.id)}):
                INDEX.insert(doc)  # type: ignore
            out = {"status": "success", "message": "Document ingested.", "chunks": 1}
            run.add_outputs(out)
            return out
        except HTTPException:
            raise
        except Exception as e:
            logging.exception("Ingest failed")
            run.add_outputs({"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

# -------- PDF ingest with page-splitting + chunking --------
@traceable
@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    with trace("ingest_pdf", run_type="chain",
               inputs={"filename": file.filename},
               tags=["ingest", "pdf"]) as run:
        try:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed")

            contents = await file.read()
            pages = extract_pdf_texts(contents)
            docs = build_page_documents(pages, file.filename)
            if not docs:
                raise HTTPException(status_code=422, detail="No text extracted from PDF")

            with trace("chunking", run_type="tool",
                       inputs={"num_pages": len(docs), "chunk_size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP}) as t:
                nodes = chunk_documents(docs)
                t.add_outputs({"num_nodes": len(nodes)})

            with trace("index.insert_nodes", run_type="retriever",
                       inputs={"num_nodes": len(nodes)}):
                INDEX.insert_nodes(nodes)  # type: ignore

            chars_ingested = sum(len(d.text) for d in docs)
            out = {
                "status": "success",
                "filename": file.filename,
                "pages_ingested": len(docs),
                "chunks_ingested": len(nodes),
                "chars_ingested": chars_ingested,
            }
            run.add_outputs(out)
            return out

        except HTTPException:
            raise
        except Exception as e:
            logging.exception("PDF ingest failed")
            run.add_outputs({"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

# -------- Query --------
@traceable
@app.post("/query")
def query_doc(req: QueryRequest):
    with trace("query_doc", run_type="chain",
               inputs={"question": req.question},
               tags=["query"]) as run:
        try:
            if not req.question.strip():
                raise HTTPException(status_code=422, detail="question is required")

            with trace("INDEX.as_query_engine", run_type="tool"):
                qe = INDEX.as_query_engine()  # type: ignore

            with trace("qe.query", run_type="llm", inputs={"question": req.question}) as sub:
                resp = qe.query(req.question)
                answer = str(resp)

                # Optional: capture brief preview + source metadata
                sources = []
                try:
                    for sn in getattr(resp, "source_nodes", [])[:10]:
                        meta = getattr(getattr(sn, "node", None), "metadata", {}) or {}
                        sources.append({
                            "node_id": getattr(sn.node, "node_id", None) if getattr(sn, "node", None) else None,
                            "score": getattr(sn, "score", None),
                            "filename": meta.get("filename"),
                            "page_number": meta.get("page_number"),
                        })
                except Exception:
                    pass

                sub.add_outputs({
                    "answer_preview": answer[:500],
                    "num_sources": len(sources),
                })

            payload = {"answer": answer, "sources": sources}
            run.add_outputs(payload)
            return payload

        except HTTPException:
            raise
        except Exception as e:
            logging.exception("Query failed")
            run.add_outputs({"error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
