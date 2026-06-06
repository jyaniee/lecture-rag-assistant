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
DEDUP_BY_PARENT = os.getenv("DEDUP_BY_PARENT", "true").lower() in ("1", "true", "yes")
MIN_PARENT_CONTENT_CHARS = int(os.getenv("MIN_PARENT_CONTENT_CHARS", "40"))
MIN_CONTEXT_TOTAL_CHARS = int(os.getenv("MIN_CONTEXT_TOTAL_CHARS", "80"))
ENABLE_QUERY_PREPROCESS = os.getenv("ENABLE_QUERY_PREPROCESS", "true").lower() in (
    "1",
    "true",
    "yes",
)
ENABLE_AMBIGUOUS_QUERY_GUIDANCE = os.getenv(
    "ENABLE_AMBIGUOUS_QUERY_GUIDANCE", "true"
).lower() in ("1", "true", "yes")
ENABLE_THIN_CONTEXT_GUARD = os.getenv("ENABLE_THIN_CONTEXT_GUARD", "true").lower() in (
    "1",
    "true",
    "yes",
)

# 쿼리 전처리 규칙 (도메인 예외 사전 대신 일반 규칙·튜닝 파라미터)
LATIN_ACRONYM_MIN_LEN = int(os.getenv("LATIN_ACRONYM_MIN_LEN", "2"))
LATIN_ACRONYM_MAX_LEN = int(os.getenv("LATIN_ACRONYM_MAX_LEN", "6"))
QUERY_BOUND_NOUNS = os.getenv(
    "QUERY_BOUND_NOUNS",
    "",
)
AMBIGUOUS_QUERY_MAX_LEN = int(os.getenv("AMBIGUOUS_QUERY_MAX_LEN", "2"))
AMBIGUOUS_SINGLE_TOKEN_MAX_LEN = int(os.getenv("AMBIGUOUS_SINGLE_TOKEN_MAX_LEN", "2"))

# 형태소 분석기 (옵션) — 과목/도메인에 무관한 토큰화 품질 개선용
ENABLE_MORPH_ANALYZER = os.getenv("ENABLE_MORPH_ANALYZER", "true").lower() in (
    "1",
    "true",
    "yes",
)
MORPH_ANALYZER = os.getenv("MORPH_ANALYZER", "kiwi").lower()
MORPH_MIN_TOKEN_LEN = int(os.getenv("MORPH_MIN_TOKEN_LEN", "2"))

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
