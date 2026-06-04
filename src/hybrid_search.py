"""BM25 + 벡터 검색 RRF 융합."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import ENABLE_HYBRID_SEARCH, RRF_K
from src.vector_store import get_indexed_chunks_for_search, invalidate_search_caches


_bm25_cache: Dict[str, Any] = {}


def _cache_key(subject: Optional[str], file_name: Optional[str]) -> str:
    return f"{subject or ''}|{file_name or ''}"


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\w가-힣]+", text) if len(t) > 1]


def _build_bm25_index(
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[BM25Okapi, List[Document]]:
    chunks = get_indexed_chunks_for_search(subject=subject, file_name=file_name)
    if not chunks:
        raise ValueError("empty_index")

    corpus = [_tokenize(doc.page_content) for doc in chunks]
    return BM25Okapi(corpus), chunks


def _get_bm25_index(
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[BM25Okapi, List[Document]]:
    key = _cache_key(subject, file_name)
    if key not in _bm25_cache:
        _bm25_cache[key] = _build_bm25_index(subject=subject, file_name=file_name)
    return _bm25_cache[key]


def clear_bm25_cache() -> None:
    _bm25_cache.clear()


def bm25_search(
    query: str,
    k: int,
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """BM25 점수(높을수록 유사)."""
    bm25, docs = _get_bm25_index(subject=subject, file_name=file_name)
    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(
        zip(docs, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(doc, float(score)) for doc, score in ranked[:k] if score > 0]


def _doc_key(doc: Document) -> str:
    meta = doc.metadata or {}
    page = meta.get("page")
    return "|".join(
        [
            str(meta.get("file_path") or meta.get("source") or ""),
            str(meta.get("file_name") or ""),
            str(page),
            doc.page_content[:80],
        ]
    )


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    *,
    rrf_k: int = RRF_K,
    final_k: int,
) -> List[Tuple[Document, float]]:
    """여러 순위 목록을 RRF 점수(낮을수록 좋음, L2와 동일 방향)로 합칩니다."""
    fused: Dict[str, Dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked):
            key = _doc_key(doc)
            if key not in fused:
                fused[key] = {"doc": doc, "rrf": 0.0}
            fused[key]["rrf"] += 1.0 / (rrf_k + rank + 1)

    merged = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:final_k]
    return [(item["doc"], 1.0 / (item["rrf"] + 1e-6)) for item in merged]


def hybrid_search_with_score(
    query: str,
    k: int,
    vector_results: List[Tuple[Document, float]],
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    if not ENABLE_HYBRID_SEARCH:
        return vector_results

    try:
        bm25_results = bm25_search(query, k, subject=subject, file_name=file_name)
    except ValueError:
        return vector_results

    if not bm25_results:
        return vector_results

    return reciprocal_rank_fusion(
        [vector_results, bm25_results],
        final_k=k,
    )


def invalidate_hybrid_cache() -> None:
    clear_bm25_cache()
    invalidate_search_caches()
