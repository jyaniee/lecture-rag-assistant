"""RRF·키워드 겹침 기반 재정렬 (추가 API 없음)."""

from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document

from src.config import ENABLE_RERANK


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[\w가-힣]+", text) if len(t) > 1}


def rerank_by_query_overlap(
    query: str,
    results: List[Tuple[Document, float]],
    top_n: int,
) -> List[Tuple[Document, float]]:
    """벡터/RRF 순위에 질문 토큰 겹침을 반영해 재정렬합니다."""
    if not ENABLE_RERANK or not results:
        return results[:top_n]

    query_tokens = _tokenize(query)
    scored: List[Tuple[Document, float, float, float]] = []

    for rank, (doc, vec_score) in enumerate(results):
        doc_tokens = _tokenize(doc.page_content)
        overlap = len(query_tokens & doc_tokens) if query_tokens else 0
        overlap_norm = overlap / max(len(query_tokens), 1)
        combined = overlap_norm * 2.0 - vec_score - rank * 0.001
        scored.append((doc, vec_score, overlap_norm, combined))

    scored.sort(key=lambda x: x[3], reverse=True)
    return [(doc, vec_score) for doc, vec_score, _, _ in scored[:top_n]]
