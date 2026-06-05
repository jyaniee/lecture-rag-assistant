"""
한국어 토큰화 유틸.

- 기본: 정규식 기반(의존성 없음)
- 옵션: 형태소 분석기(현재 Kiwi) 기반 토큰화

목표는 "과목/도메인에 무관하게" BM25·키워드 기반 검색 토큰 품질을 개선하는 것입니다.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

from src.config import ENABLE_MORPH_ANALYZER, MORPH_ANALYZER, MORPH_MIN_TOKEN_LEN


_RE_TOKEN = re.compile(r"[\w가-힣]+")


def _regex_tokens(text: str) -> List[str]:
    return [t.lower() for t in _RE_TOKEN.findall(text or "") if len(t) >= MORPH_MIN_TOKEN_LEN]


@lru_cache(maxsize=1)
def _get_kiwi():
    from kiwipiepy import Kiwi  # type: ignore

    return Kiwi()


def _kiwi_tokens(text: str) -> List[str]:
    kiwi = _get_kiwi()
    tokens: List[str] = []
    for token in kiwi.tokenize(text or ""):
        form = getattr(token, "form", "") or ""
        tag = getattr(token, "tag", "") or ""
        if not form:
            continue

        # 품질/일반성 균형: 기능어(조사/어미 등)는 대체로 검색에 불리
        if tag.startswith(("J", "E", "S")):
            continue

        form = form.lower()
        if len(form) < MORPH_MIN_TOKEN_LEN:
            continue

        tokens.append(form)

    return tokens


def tokenize(text: str) -> List[str]:
    """
    BM25 등 키워드 검색용 토큰.

    - 형태소 분석기를 켰고(ENABLE_MORPH_ANALYZER) 사용 가능하면 형태소 기반 토큰을 사용
    - 실패 시 정규식 토큰으로 자동 폴백
    """

    if not ENABLE_MORPH_ANALYZER:
        return _regex_tokens(text)

    if MORPH_ANALYZER != "kiwi":
        return _regex_tokens(text)

    try:
        return _kiwi_tokens(text)
    except Exception:
        return _regex_tokens(text)

