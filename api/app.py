import os, logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.groq import Groq

import psycopg2  # pip install psycopg2-binary

# -----------------------------
# Env
# -----------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "llamadb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

PGVECTOR_SCHEMA = os.getenv("PGVECTOR_SCHEMA", "public")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "data_llama_index")  # match what errors reference

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

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
        return {"provider": "openai" if OPENAI_API_KEY else "huggingface", "model": model_name, "embed_dim": dim}
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
            # 1) extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # 2) schema
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{PGVECTOR_SCHEMA}";')
            # 3) table (columns expected by LlamaIndex PGVectorStore)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS "{PGVECTOR_SCHEMA}"."{PGVECTOR_TABLE}" (
                  id BIGSERIAL PRIMARY KEY,
                  node_id TEXT,
                  text TEXT,
                  metadata_ JSONB,
                  embedding VECTOR({embed_dim})
                );
            """)
            # 4) helpful indexes
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
    # Make sure the DB objects exist before PGVectorStore touches them
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
# FastAPI
# -----------------------------
app = FastAPI(title="Llama API")
INDEX: Optional[VectorStoreIndex] = None
STARTUP_INFO = {}

@app.on_event("startup")
def on_startup():
    global INDEX, STARTUP_INFO
    emb = init_embeddings()
    llm = init_llm()
    INDEX = init_vector_index(emb["embed_dim"], llm)
    STARTUP_INFO = {"embeddings": emb, "llm": {"provider": "groq", "model": llm.model},
                    "pgvector": {"schema": PGVECTOR_SCHEMA, "table": PGVECTOR_TABLE}}

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

@app.post("/ingest")
def ingest_doc(req: IngestRequest):
    try:
        if not req.text.strip():
            raise HTTPException(status_code=422, detail="text is required")
        doc = Document(text=req.text, doc_id=req.id) if req.id else Document(text=req.text)
        INDEX.insert(doc)  # type: ignore
        return {"status": "success", "message": "Document ingested."}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_doc(req: QueryRequest):
    try:
        if not req.question.strip():
            raise HTTPException(status_code=422, detail="question is required")
        qe = INDEX.as_query_engine()  # type: ignore
        resp = qe.query(req.question)
        return {"answer": str(resp)}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
