from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    ENABLE_HYBRID_SEARCH,
    ENABLE_RERANK,
    RERANK_FETCH_MULTIPLIER,
    RETRIEVER_TOP_K,
)
from src.hybrid_search import hybrid_search_with_score
from src.rerank import rerank_by_query_overlap
from src.vector_store import load_vector_store


def get_retriever() -> VectorStoreRetriever:
    """
    Chroma DB에서 질문과 유사한 문서 청크를 검색하는 Retriever를 생성합니다.
    """
    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )


def build_metadata_filter(
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Chroma where 필터 dict를 만듭니다. loaders 메타 키와 동일하게 사용합니다."""
    if subject and file_name:
        return {"$and": [{"subject": subject}, {"file_name": file_name}]}
    if subject:
        return {"subject": subject}
    if file_name:
        return {"file_name": file_name}
    return None


def _doc_lookup_key(doc: Document) -> str:
    meta = doc.metadata or {}
    page = meta.get("page")
    return "|".join(
        [
            str(meta.get("file_path") or meta.get("source") or ""),
            str(meta.get("file_name") or ""),
            str(page),
            (doc.page_content or "")[:120],
        ]
    )


def search_vector_with_score(
    query: str,
    k: int,
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """벡터 유사도만 (L2, 낮을수록 유사). threshold 판정용."""
    vector_store = load_vector_store()
    metadata_filter = build_metadata_filter(subject=subject, file_name=file_name)

    kwargs: Dict[str, Any] = {"k": k}
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter

    return vector_store.similarity_search_with_score(query=query, **kwargs)


def _attach_vector_scores(
    ranked: List[Tuple[Document, float]],
    vector_score_map: Dict[str, float],
    fallback_score: float,
) -> List[Tuple[Document, float]]:
    """융합·재정렬 결과에 L2 점수를 붙입니다 (threshold 필터 호환)."""
    attached: List[Tuple[Document, float]] = []
    for doc, _ in ranked:
        score = vector_score_map.get(_doc_lookup_key(doc), fallback_score)
        attached.append((doc, score))
    return attached


def search_documents_with_score(
    query: str,
    k: Optional[int] = None,
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    질문과 유사한 문서 청크를 점수(L2)와 함께 검색합니다.

    하이브리드·재정렬 활성 시 fetch를 넓힌 뒤 RRF·겹침 재정렬합니다.
    threshold는 원본 벡터 점수 맵으로 필터합니다.
    """
    search_k = k if k is not None else RETRIEVER_TOP_K
    use_enhanced = ENABLE_HYBRID_SEARCH or ENABLE_RERANK
    fetch_k = (
        min(search_k * max(RERANK_FETCH_MULTIPLIER, 1), 24)
        if use_enhanced
        else search_k
    )

    vector_results = search_vector_with_score(
        query,
        fetch_k,
        subject=subject,
        file_name=file_name,
    )
    if not vector_results:
        return []

    vector_score_map = {_doc_lookup_key(doc): score for doc, score in vector_results}
    best_vector = vector_results[0][1]

    ranked = vector_results
    if ENABLE_HYBRID_SEARCH:
        ranked = hybrid_search_with_score(
            query,
            fetch_k,
            vector_results,
            subject=subject,
            file_name=file_name,
        )

    if ENABLE_RERANK:
        ranked = rerank_by_query_overlap(query, ranked, top_n=fetch_k)

    ranked = ranked[:search_k]
    return _attach_vector_scores(ranked, vector_score_map, best_vector)
