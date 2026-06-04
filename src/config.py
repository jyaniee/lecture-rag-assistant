import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
RELEVANCE_SCORE_THRESHOLD = float(os.getenv("RELEVANCE_SCORE_THRESHOLD", "1.2"))

DATA_DIR = os.getenv("DATA_DIR", "data/raw")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
CHROMA_DIR = os.getenv("CHROMA_DIR", "vectorstore/chroma")

MAX_CHAT_HISTORY_TURNS = int(os.getenv("MAX_CHAT_HISTORY_TURNS", "3"))
CHAT_HISTORY_MAX_CHARS = int(os.getenv("CHAT_HISTORY_MAX_CHARS", "400"))
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "true").lower() in ("1", "true", "yes")
FOLLOW_UP_SEARCH_K_MULTIPLIER = int(os.getenv("FOLLOW_UP_SEARCH_K_MULTIPLIER", "2"))

# --- 추천도 2 (고급 RAG) ---
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() in ("1", "true", "yes")
RERANK_FETCH_MULTIPLIER = int(os.getenv("RERANK_FETCH_MULTIPLIER", "3"))
RRF_K = int(os.getenv("RRF_K", "60"))

ENABLE_CONTEXT_COMPRESS = os.getenv("ENABLE_CONTEXT_COMPRESS", "true").lower() in ("1", "true", "yes")
CONTEXT_MAX_CHARS = int(os.getenv("CONTEXT_MAX_CHARS", "2800"))

ENABLE_PARENT_CHILD = os.getenv("ENABLE_PARENT_CHILD", "true").lower() in ("1", "true", "yes")
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "450"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "80"))
PARENT_CONTENT_MAX_CHARS = int(os.getenv("PARENT_CONTENT_MAX_CHARS", "6000"))

ENABLE_LLM_STREAMING = os.getenv("ENABLE_LLM_STREAMING", "true").lower() in ("1", "true", "yes")
