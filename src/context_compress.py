"""검색 청크·parent 텍스트의 extractive 압축 (LLM 없음)."""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from src.config import CONTEXT_MAX_CHARS, ENABLE_CONTEXT_COMPRESS


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[\w가-힣]+", text) if len(t) > 1]


def _sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?。\n])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def compress_text_for_query(
    query: str,
    text: str,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """질문과 겹치는 문장 위주로 텍스트를 줄입니다."""
    text = text.strip()
    if not text or len(text) <= max_chars:
        return text

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return text[:max_chars].rstrip() + "..."

    sentences = _sentence_split(text)
    if not sentences:
        return text[:max_chars].rstrip() + "..."

    scored: List[tuple[int, int, str]] = []
    for idx, sentence in enumerate(sentences):
        tokens = set(_tokenize(sentence))
        overlap = len(query_tokens & tokens)
        scored.append((overlap, -idx, sentence))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    chosen: List[str] = []
    total = 0
    for _, _, sentence in scored:
        add_len = len(sentence) + (2 if chosen else 0)
        if total + add_len > max_chars:
            continue
        chosen.append(sentence)
        total += add_len

    if not chosen:
        return text[:max_chars].rstrip() + "..."

    chosen.sort(key=lambda s: text.find(s))
    return "\n".join(chosen)


def compress_document_for_query(
    query: str,
    doc: Document,
    *,
    max_chars: int = CONTEXT_MAX_CHARS,
) -> Document:
    if not ENABLE_CONTEXT_COMPRESS:
        return doc

    compressed = compress_text_for_query(query, doc.page_content, max_chars=max_chars)
    if compressed == doc.page_content:
        return doc

    return Document(page_content=compressed, metadata=dict(doc.metadata))
