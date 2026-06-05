"""
질문·검색 쿼리 전처리·정규화.

도메인별 예외 사전 대신, 아래와 같은 일반 규칙을 적용합니다.
- 라틴 약어: 전부 소문자인 짧은 토큰 → 대문자 (일반 영어 단어는 제외)
- 한글: 조사 경계·한글↔라틴 경계·복합 명사 꼬리(설정 가능) 분리
- 모호 질문: 질문 의도 신호가 없고 극단적으로 짧을 때만 (검색 전 차단)

검색(임베딩) 직전에 적용합니다. 사용자 원문 질문은 LLM에 그대로 전달할 수 있습니다.
"""

from __future__ import annotations

import re
from typing import List, Set

from langchain_core.documents import Document

from src.config import (
    AMBIGUOUS_QUERY_MAX_LEN,
    AMBIGUOUS_SINGLE_TOKEN_MAX_LEN,
    LATIN_ACRONYM_MAX_LEN,
    LATIN_ACRONYM_MIN_LEN,
    MIN_CONTEXT_TOTAL_CHARS,
    MIN_PARENT_CONTENT_CHARS,
    QUERY_BOUND_NOUNS,
)

# 일반 영어 기능어 — 약어 대문자화에서 제외 (도메인 약어 목록이 아님)
_LATIN_STOPWORDS: Set[str] = frozenset(
    {
        "a",
        "an",
        "as",
        "at",
        "be",
        "by",
        "do",
        "go",
        "he",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "no",
        "of",
        "on",
        "or",
        "so",
        "to",
        "up",
        "us",
        "we",
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "how",
        "its",
        "may",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "why",
    }
)

# 질문 의도가 드러나는 표현 — 있으면 모호 질문으로 보지 않음
_QUESTION_INTENT = re.compile(
    r"[?？]"
    r"|(?:왜|어떻|무엇|뭐|무슨|언제|어디|누구|어느|몇)"
    r"|(?:설명|알려|정리|요약|비교|차이|원리|이유|방법|뜻|의미|정의|증명|예시)",
)

# 목적격·부사격 등 길이 1 초과 조사만 분리 (은/는/이/가는 동사·형용사 어미와 혼동)
_PARTICLE_BOUNDARY = re.compile(
    r"([가-힣]{2,})(을|를|에서|으로|에게|께|한테)([가-힣])"
)

# 한글 어휘(2자+) ↔ 라틴 토큰(2자+) 경계만 분리 (RAG가·rag가 등 조사 붙임 유지)
_HANGUL_TO_LATIN = re.compile(r"([가-힣]{2,})([A-Za-z]{2,})")
_LATIN_TO_HANGUL = re.compile(r"([A-Za-z]{2,})([가-힣]{2,})")

_LATIN_TOKEN = re.compile(
    rf"(?<![A-Za-z])([a-z]{{{LATIN_ACRONYM_MIN_LEN},{LATIN_ACRONYM_MAX_LEN}}})(?![A-Za-z])"
)

_AMBIGUOUS_GUIDANCE = (
    "질문이 너무 짧거나 모호해 강의자료 검색이 어렵습니다. "
    "궁금한 개념과 알고 싶은 점을 한 문장으로 구체적으로 적어 주세요."
)

_THIN_CONTEXT_GUIDANCE = (
    "제공된 강의자료에서 답변에 쓸 만한 본문을 찾지 못했습니다. "
    "질문을 더 구체적으로 하거나 다른 표현으로 다시 시도해 주세요."
)


def _bound_noun_pattern() -> re.Pattern[str] | None:
    """설정(QUERY_BOUND_NOUNS)에서 복합 명사 꼬리 패턴을 만듭니다."""
    nouns = [n.strip() for n in QUERY_BOUND_NOUNS.split(",") if n.strip()]
    if not nouns:
        return None
    joined = "|".join(re.escape(n) for n in nouns)
    return re.compile(rf"([가-힣]{{2,}})({joined})([가-힣?]?)")


def normalize_latin_acronyms(text: str) -> str:
    """전부 소문자인 짧은 라틴 토큰을 약어로 보고 대문자화합니다."""

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token in _LATIN_STOPWORDS:
            return token
        return token.upper()

    return _LATIN_TOKEN.sub(_replace, text)


def normalize_korean_spacing(text: str) -> str:
    """조사·한글-라틴 경계·복합 명사 꼬리를 규칙 기반으로 분리합니다."""
    text = _PARTICLE_BOUNDARY.sub(r"\1\2 \3", text)
    text = _HANGUL_TO_LATIN.sub(r"\1 \2", text)
    text = _LATIN_TO_HANGUL.sub(r"\1 \2", text)

    bound_noun = _bound_noun_pattern()
    if bound_noun:
        text = bound_noun.sub(r"\1 \2\3", text)

    return text


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def preprocess_for_search(query: str) -> str:
    """검색·임베딩 직전 쿼리 정규화."""
    if not query:
        return query

    text = normalize_whitespace(query)
    # 약어 대문자화를 먼저 적용해 RAG가·rag가 같은 붙임 표기를 유지합니다.
    text = normalize_latin_acronyms(text)
    text = normalize_korean_spacing(text)
    return normalize_whitespace(text)


def _tokenize(query: str) -> List[str]:
    return re.findall(r"[\w가-힣]+", query)


def has_question_intent(question: str) -> bool:
    return bool(_QUESTION_INTENT.search(question))


def is_ambiguous_query(question: str) -> bool:
    """
    검색 전 차단은 최소한만 적용합니다.
    질문 의도가 보이면 통과하고, 빈 입력·극단적으로 짧은 경우만 모호로 봅니다.
    (짧은 약어 질문은 전처리 후 검색·threshold로 판단)
    """
    q = normalize_whitespace(question)
    if not q:
        return True

    if has_question_intent(q):
        return False

    if len(q) <= AMBIGUOUS_QUERY_MAX_LEN:
        return True

    tokens = _tokenize(q)
    if len(tokens) == 1 and len(tokens[0]) <= AMBIGUOUS_SINGLE_TOKEN_MAX_LEN:
        return True

    return False


def ambiguous_query_message() -> str:
    return _AMBIGUOUS_GUIDANCE


def thin_context_message() -> str:
    return _THIN_CONTEXT_GUIDANCE


def _rich_text_length(doc: Document) -> int:
    meta = doc.metadata or {}
    text = meta.get("parent_content") or doc.page_content or ""
    return len(str(text).strip())


def assess_context_richness(docs: List[Document]) -> dict:
    """LLM 투입 전 context 풍부도."""
    if not docs:
        return {
            "total_chars": 0,
            "rich_doc_count": 0,
            "is_sufficient": False,
        }

    lengths = [_rich_text_length(d) for d in docs]
    rich_count = sum(1 for n in lengths if n >= MIN_PARENT_CONTENT_CHARS)
    total = sum(lengths)

    return {
        "total_chars": total,
        "rich_doc_count": rich_count,
        "is_sufficient": total >= MIN_CONTEXT_TOTAL_CHARS and rich_count >= 1,
    }
