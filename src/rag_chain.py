import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import (
    DEDUP_BY_PARENT,
    ENABLE_AMBIGUOUS_QUERY_GUIDANCE,
    ENABLE_CONTEXT_COMPRESS,
    ENABLE_HYBRID_SEARCH,
    ENABLE_LLM_STREAMING,
    ENABLE_PARENT_CHILD,
    ENABLE_QUERY_PREPROCESS,
    ENABLE_QUERY_REWRITE,
    ENABLE_RERANK,
    ENABLE_THIN_CONTEXT_GUARD,
    FOLLOW_UP_SEARCH_K_MULTIPLIER,
    MIN_PARENT_CONTENT_CHARS,
    RELEVANCE_SCORE_THRESHOLD,
    RETRIEVER_TOP_K,
)
from src.context_compress import compress_document_for_query
from src.generator import build_prompt, format_chat_history_block, get_llm
from src.query_preprocess import (
    ambiguous_query_message,
    assess_context_richness,
    is_ambiguous_query,
    preprocess_for_search,
    thin_context_message,
)
from src.retriever import search_documents_with_score


def is_low_relevance(
    search_results: List[Tuple[Document, float]],
    threshold: float = RELEVANCE_SCORE_THRESHOLD,
) -> bool:
    """검색 결과의 관련도가 낮은지 판단합니다."""
    if not search_results:
        return True

    best_score = search_results[0][1]

    return best_score > threshold


def filter_relevant_documents(
    search_results: List[Tuple[Document, float]],
    threshold: float = RELEVANCE_SCORE_THRESHOLD,
) -> List[Document]:
    """threshold 이하 점수를 가진 문서 청크만 반환합니다."""
    return [doc for doc, score in search_results if score <= threshold]


def filter_relevant_doc_score_pairs(
    search_results: List[Tuple[Document, float]],
    threshold: float = RELEVANCE_SCORE_THRESHOLD,
) -> List[Tuple[Document, float]]:
    """threshold 이하 (문서, score) 쌍만 반환합니다."""
    return [(doc, score) for doc, score in search_results if score <= threshold]


PIPELINE_REJECTION_ANSWER = (
    "제공된 강의자료에서 질문과 관련된 내용을 찾기 어렵습니다."
)

SUBJECT_REQUIRED_ANSWER = (
    "검색할 과목이 지정되지 않았습니다. 과목을 선택한 뒤 다시 질문해 주세요."
)


def _parent_dedup_key(doc: Document) -> str:
    meta = doc.metadata or {}
    parent_id = meta.get("parent_id")
    if parent_id:
        return str(parent_id)
    file_name = meta.get("file_name") or meta.get("source") or "unknown"
    page = meta.get("page")
    return f"{file_name}|{page}"


def _parent_content_length(doc: Document) -> int:
    meta = doc.metadata or {}
    text = meta.get("parent_content") or doc.page_content or ""
    return len(str(text).strip())


def deduplicate_by_file(
    doc_score_pairs: List[Tuple[Document, float]],
) -> List[Document]:
    """파일당 가장 유사한(낮은 score) 청크 하나만 남깁니다."""
    best_per_file: Dict[str, Tuple[Document, float]] = {}

    for doc, score in doc_score_pairs:
        key = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"
        if key not in best_per_file or score < best_per_file[key][1]:
            best_per_file[key] = (doc, score)

    return [
        doc for doc, _ in sorted(best_per_file.values(), key=lambda item: item[1])
    ]


def deduplicate_for_context(
    doc_score_pairs: List[Tuple[Document, float]],
    *,
    max_docs: Optional[int] = None,
) -> List[Document]:
    """
    context용 dedup.

    DEDUP_BY_PARENT=true: 페이지(parent)당 best 1청크 — 같은 PDF 여러 슬라이드 허용.
    짧은 제목-only parent는 풍부한 parent가 있으면 뒤로 밀거나 제외.
    """
    limit = max_docs if max_docs is not None else RETRIEVER_TOP_K

    if DEDUP_BY_PARENT:
        best_per_parent: Dict[str, Tuple[Document, float]] = {}
        for doc, score in doc_score_pairs:
            key = _parent_dedup_key(doc)
            if key not in best_per_parent or score < best_per_parent[key][1]:
                best_per_parent[key] = (doc, score)

        ranked = sorted(
            best_per_parent.values(),
            key=lambda item: (item[1], -_parent_content_length(item[0])),
        )
        rich = [
            (doc, score)
            for doc, score in ranked
            if _parent_content_length(doc) >= MIN_PARENT_CONTENT_CHARS
        ]
        if rich:
            ranked = rich + [
                (doc, score)
                for doc, score in ranked
                if _parent_content_length(doc) < MIN_PARENT_CONTENT_CHARS
            ]

        return [doc for doc, _ in ranked[:limit]]

    return deduplicate_by_file(doc_score_pairs)[:limit]


def normalize_chat_history(
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """user/assistant 메시지만 정규화합니다."""
    if not chat_history:
        return []

    normalized: List[Dict[str, Any]] = []
    for msg in chat_history:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    return normalized


def _assistant_was_rejected(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if text.startswith(PIPELINE_REJECTION_ANSWER):
        return True
    if "제공된 강의자료에서" in text and "찾기 어렵습니다" in text:
        return True
    return False


def _is_standalone_question(question: str) -> bool:
    """완결된 질문이면 query rewrite 대상에서 제외합니다."""
    q = question.strip()
    if len(q) > 60:
        return True
    if re.search(r"[?？]$", q):
        return True
    if re.search(
        r"(설명|알려|정리|요약|차이|원리|이유|방법|뜻|의미|무엇|뭐야|어떻|왜|언제|증명|비교)",
        q,
    ):
        return True
    if len(re.findall(r"[\w가-힣]+", q)) >= 5:
        return True
    return False


def _find_last_user_for_rewrite(history: List[Dict[str, Any]]) -> Optional[str]:
    """거부된 턴의 user 질문은 rewrite 맥락에서 제외합니다."""
    for idx in range(len(history) - 1, -1, -1):
        msg = history[idx]
        if msg["role"] != "user":
            continue

        next_msg = history[idx + 1] if idx + 1 < len(history) else None
        if next_msg and next_msg["role"] == "assistant":
            if _assistant_was_rejected(next_msg["content"]):
                continue

        return msg["content"]

    return None


def build_search_query(
    question: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    2차 RAG: 후속 질문 검색 품질 보강.
    짧고 애매한 후속만 이전 user와 합칩니다.
    """
    question = question.strip()
    history = normalize_chat_history(chat_history)

    if not ENABLE_QUERY_REWRITE or not history:
        return _finalize_search_query(question)

    if _is_standalone_question(question):
        return _finalize_search_query(question)

    last_user = _find_last_user_for_rewrite(history)
    if not last_user or last_user == question:
        return _finalize_search_query(question)

    if len(question) <= 60:
        merged = f"{last_user} {question}".strip()
        return _finalize_search_query(merged)

    return _finalize_search_query(question)


def _finalize_search_query(raw_query: str) -> str:
    if not ENABLE_QUERY_PREPROCESS:
        return raw_query.strip()
    return preprocess_for_search(raw_query)


def _effective_search_k(
    base_k: int,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """후속 질문 시 검색 후보를 넓힙니다 (2차 RAG)."""
    history = normalize_chat_history(chat_history)
    if not history:
        return base_k

    return min(base_k * max(FOLLOW_UP_SEARCH_K_MULTIPLIER, 1), 12)


def _resolve_rejection(
    search_results: List[Tuple[Document, float]],
    threshold: float,
    *,
    max_context_docs: Optional[int] = None,
) -> tuple[List[Document], Optional[str]]:
    """검색 결과에서 LLM에 넣을 문서와 거부 사유를 결정합니다."""
    if not search_results:
        return [], "empty_index"

    if is_low_relevance(search_results, threshold=threshold):
        return [], "low_best_score"

    filtered_pairs = filter_relevant_doc_score_pairs(search_results, threshold=threshold)
    if not filtered_pairs:
        return [], "no_chunks_passed_filter"

    docs = deduplicate_for_context(
        filtered_pairs,
        max_docs=max_context_docs,
    )
    if not docs:
        return [], "no_chunks_passed_filter"

    if ENABLE_THIN_CONTEXT_GUARD:
        richness = assess_context_richness(docs)
        if not richness["is_sufficient"]:
            return [], "thin_context"

    return docs, None


def run_retrieval(
    question: str,
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
    top_k: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    LLM 없이 검색·필터·dedup만 수행합니다. 튜닝·품질 실험용.
    """
    effective_k = top_k if top_k is not None else RETRIEVER_TOP_K
    threshold = (
        relevance_threshold
        if relevance_threshold is not None
        else RELEVANCE_SCORE_THRESHOLD
    )
    subject_key = (subject or "").strip()
    if not subject_key:
        retrieval = build_retrieval_meta(
            top_k=effective_k,
            threshold=threshold,
            search_results=[],
            context_docs=[],
            rejection_reason="subject_required",
            subject=None,
        )
        return {
            "is_rejected": True,
            "rejection_reason": "subject_required",
            "context_docs": [],
            "debug_scores": [],
            "retrieval": retrieval,
        }

    history = normalize_chat_history(chat_history)
    search_query_raw = build_search_query(question, history)
    search_query = search_query_raw

    if ENABLE_AMBIGUOUS_QUERY_GUIDANCE and is_ambiguous_query(question):
        retrieval = build_retrieval_meta(
            top_k=effective_k,
            threshold=threshold,
            search_results=[],
            context_docs=[],
            rejection_reason="ambiguous_query",
            search_query=search_query,
            search_query_raw=search_query_raw,
            subject=subject_key,
        )
        return {
            "is_rejected": True,
            "rejection_reason": "ambiguous_query",
            "context_docs": [],
            "debug_scores": [],
            "retrieval": retrieval,
        }

    search_k = _effective_search_k(effective_k, history)

    search_results = search_documents_with_score(
        search_query,
        k=search_k,
        subject=subject_key,
        file_name=file_name,
    )

    context_docs, rejection_reason = _resolve_rejection(
        search_results,
        threshold,
        max_context_docs=effective_k,
    )

    debug_scores = [
        {
            "file_name": doc.metadata.get("file_name", doc.metadata.get("source", "알 수 없는 문서")),
            "page": doc.metadata.get("page"),
            "score": score,
        }
        for doc, score in search_results
    ]

    return {
        "is_rejected": rejection_reason is not None,
        "rejection_reason": rejection_reason,
        "context_docs": context_docs,
        "debug_scores": debug_scores,
        "retrieval": build_retrieval_meta(
            top_k=effective_k,
            threshold=threshold,
            search_results=search_results,
            context_docs=context_docs,
            rejection_reason=rejection_reason,
            search_query=search_query,
            search_query_raw=search_query_raw,
            search_k_used=search_k,
            context_richness=assess_context_richness(context_docs),
            subject=subject_key,
        ),
    }


def build_retrieval_meta(
    *,
    top_k: int,
    threshold: float,
    search_results: List[Tuple[Document, float]],
    context_docs: List[Document],
    rejection_reason: Optional[str],
    search_query: Optional[str] = None,
    search_query_raw: Optional[str] = None,
    search_k_used: Optional[int] = None,
    context_richness: Optional[Dict[str, Any]] = None,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    best_score = search_results[0][1] if search_results else None
    return {
        "subject": subject,
        "top_k": top_k,
        "search_k_used": search_k_used or top_k,
        "threshold": threshold,
        "raw_hit_count": len(search_results),
        "context_chunk_count": len(context_docs),
        "best_score": best_score,
        "rejection_reason": rejection_reason,
        "search_query": search_query,
        "search_query_raw": search_query_raw or search_query,
        "query_rewrite_enabled": ENABLE_QUERY_REWRITE,
        "query_preprocess_enabled": ENABLE_QUERY_PREPROCESS,
        "ambiguous_guidance_enabled": ENABLE_AMBIGUOUS_QUERY_GUIDANCE,
        "thin_context_guard_enabled": ENABLE_THIN_CONTEXT_GUARD,
        "hybrid_enabled": ENABLE_HYBRID_SEARCH,
        "rerank_enabled": ENABLE_RERANK,
        "parent_child_enabled": ENABLE_PARENT_CHILD,
        "context_compress_enabled": ENABLE_CONTEXT_COMPRESS,
        "dedup_by_parent": DEDUP_BY_PARENT,
        "context_richness": context_richness,
    }


def expand_to_parent_context(doc: Document) -> Document:
    """child 청크 검색 시 parent(페이지/슬라이드) 전문으로 LLM context를 확장합니다."""
    parent_text = (doc.metadata or {}).get("parent_content")
    if parent_text:
        return Document(page_content=str(parent_text), metadata=dict(doc.metadata))
    return doc


def prepare_context_documents(
    query: str,
    docs: List[Document],
) -> List[Document]:
    """parent 확장 + extractive 압축."""
    prepared: List[Document] = []
    seen_parent: set[str] = set()

    for doc in docs:
        expanded = expand_to_parent_context(doc)
        parent_id = (expanded.metadata or {}).get("parent_id")
        if parent_id:
            if parent_id in seen_parent:
                continue
            seen_parent.add(parent_id)

        prepared.append(compress_document_for_query(query, expanded))

    return prepared


_INLINE_SOURCE_LINE = re.compile(
    r"^\s*(?:출처|참고(?:\s*문서)?)\s*[：:]\s*.+",
    re.IGNORECASE,
)


def strip_inline_source_citations(text: str) -> str:
    """답변 본문 끝의 '출처: 참고 문서 N' 등 UI와 중복되는 인용 줄을 제거합니다."""
    if not (text or "").strip():
        return text or ""

    lines = text.rstrip().splitlines()
    while lines:
        line = lines[-1].strip()
        if not line:
            lines.pop()
            continue
        if _INLINE_SOURCE_LINE.match(line):
            lines.pop()
            continue
        if "참고" in line and "문서" in line and re.search(r"\d", line):
            lines.pop()
            continue
        break

    return "\n".join(lines).rstrip()


def clean_preview_text(text: str, max_length: int = 500) -> str:
    """UI 표시용 문서 내용을 정리합니다."""
    remove_chars = ["■", "□", "▪", "▫", "●", "○", "◆", "◇", "•", "·"]

    for char in remove_chars:
        text = text.replace(char, " ")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if len(text) > max_length:
        text = text[:max_length].rstrip() + "..."

    return text


def format_documents(
    documents: List[Document],
    query: Optional[str] = None,
) -> str:
    """
    검색된 문서를 LLM 프롬프트에 넣기 좋은 문자열로 변환합니다.
    """
    if query:
        documents = prepare_context_documents(query, documents)

    formatted_chunks = []

    for i, doc in enumerate(documents, start=1):
        page = doc.metadata.get("page", None)
        page_text = f"p.{page + 1}" if isinstance(page, int) else "페이지 정보 없음"

        formatted_chunks.append(
            f"[자료 {i}]\n"
            f"페이지: {page_text}\n"
            f"내용:\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def format_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """UI에서 출처를 표시하기 위한 구조화된 데이터로 변환합니다."""
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "알 수 없는 문서")
        page = doc.metadata.get("page", None)

        source_path = Path(source)
        subject = doc.metadata.get("subject") or source_path.parent.name
        file_name = doc.metadata.get("file_name") or source_path.name

        if not subject or subject == ".":
            subject = "과목 정보 없음"

        preview = clean_preview_text(doc.page_content)

        sources.append(
            {
                "subject": subject,
                "file_name": file_name,
                "source": source,
                "page": page + 1 if isinstance(page, int) else None,
                "content": preview,
            }
        )

    return sources


def _rejection_answer_for_reason(reason: Optional[str]) -> str:
    if reason == "subject_required":
        return SUBJECT_REQUIRED_ANSWER
    if reason == "ambiguous_query":
        return ambiguous_query_message()
    if reason == "thin_context":
        return thin_context_message()
    return PIPELINE_REJECTION_ANSWER


def _rejected_answer_dict(
    answer_mode: str,
    debug_scores: List[Dict[str, Any]],
    retrieval: Dict[str, Any],
) -> Dict[str, Any]:
    reason = retrieval.get("rejection_reason")
    return {
        "answer": _rejection_answer_for_reason(reason),
        "sources": [],
        "answer_mode": answer_mode,
        "is_rejected": True,
        "debug_scores": debug_scores,
        "retrieval": retrieval,
    }


def _invoke_llm(
    question: str,
    answer_mode: str,
    docs: List[Document],
    history: List[Dict[str, Any]],
) -> str:
    context = format_documents(docs, query=question)
    conversation = format_chat_history_block(history)
    prompt = build_prompt(answer_mode)
    chain = prompt | get_llm()
    response = chain.invoke(
        {
            "conversation": conversation,
            "question": question,
            "context": context,
        }
    )
    return strip_inline_source_citations(response.content)


def ask_question(
    question: str,
    answer_mode: str = "기본 Q&A",
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
    top_k: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    질문을 받아 RAG 방식으로 답변과 출처를 반환합니다.

    매 턴 인덱싱된 자료를 검색합니다. chat_history는 맥락·검색 쿼리 보강용입니다.
    """
    retrieved = run_retrieval(
        question,
        subject=subject,
        file_name=file_name,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        chat_history=chat_history,
    )
    debug_scores = retrieved["debug_scores"]
    retrieval = retrieved["retrieval"]

    if retrieved["is_rejected"]:
        return _rejected_answer_dict(answer_mode, debug_scores, retrieval)

    docs = retrieved["context_docs"]
    history = normalize_chat_history(chat_history)
    answer = _invoke_llm(question, answer_mode, docs, history)

    return {
        "answer": answer,
        "sources": format_sources(docs),
        "answer_mode": answer_mode,
        "is_rejected": False,
        "debug_scores": debug_scores,
        "retrieval": retrieval,
    }


def ask_question_events(
    question: str,
    answer_mode: str = "기본 Q&A",
    *,
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
    top_k: Optional[int] = None,
    relevance_threshold: Optional[float] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    LLM 스트리밍용 이벤트 제너레이터.

    - {"type": "token", "content": "..."}
    - {"type": "final", "data": ask_question과 동일 dict}
    """
    retrieved = run_retrieval(
        question,
        subject=subject,
        file_name=file_name,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
        chat_history=chat_history,
    )
    debug_scores = retrieved["debug_scores"]
    retrieval = retrieved["retrieval"]

    if retrieved["is_rejected"]:
        payload = _rejected_answer_dict(answer_mode, debug_scores, retrieval)
        yield {"type": "final", "data": payload}
        return

    docs = retrieved["context_docs"]
    history = normalize_chat_history(chat_history)

    if not ENABLE_LLM_STREAMING:
        yield {
            "type": "final",
            "data": {
                "answer": _invoke_llm(question, answer_mode, docs, history),
                "sources": format_sources(docs),
                "answer_mode": answer_mode,
                "is_rejected": False,
                "debug_scores": debug_scores,
                "retrieval": retrieval,
            },
        }
        return

    context = format_documents(docs, query=question)
    conversation = format_chat_history_block(history)
    chain = build_prompt(answer_mode) | get_llm()

    parts: List[str] = []
    for chunk in chain.stream(
        {
            "conversation": conversation,
            "question": question,
            "context": context,
        }
    ):
        text = getattr(chunk, "content", None) or ""
        if text:
            parts.append(text)
            yield {"type": "token", "content": text}

    yield {
        "type": "final",
        "data": {
            "answer": strip_inline_source_citations("".join(parts)),
            "sources": format_sources(docs),
            "answer_mode": answer_mode,
            "is_rejected": False,
            "debug_scores": debug_scores,
            "retrieval": retrieval,
        },
    }


