from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import CHAT_HISTORY_MAX_CHARS, MAX_CHAT_HISTORY_TURNS, OPENAI_CHAT_MODEL


BASE_RULES = """
당신은 강의자료 기반 학습 도우미입니다.

공통 규칙:
1. 반드시 제공된 참고 문서 내용을 우선 근거로 답변하세요.
2. 참고 문서에 직접적인 정의문이 없더라도, 문서에 등장하는 키워드·설명·예시 범위 안에서만 정리할 수 있으면 답변하세요. 문서에 없는 사실, 수치, 용어 정의는 추가하지 마세요.
3. 참고 문서와 전혀 무관한 내용은 추측하지 말고 "제공된 자료만으로는 확인하기 어렵습니다"라고 답변하세요.
4. 답변은 한국어로 작성하세요.
5. 개념 질문에는 먼저 쉬운 정의를 제시한 뒤, 작동 방식이나 활용 맥락을 설명하세요.
6. 참고 문서에 포함된 강의 제목, 슬라이드 제목, 파일명, 문서명은 개념의 정의로 사용하지 마세요.
7. 답변 본문에는 파일명을 억지로 언급하지 마세요. 출처는 별도 참고 자료 영역에서 표시됩니다.
8. 이전 대화는 질문 맥락 이해용입니다. 사실·정의·수치는 반드시 이번 턴의 참고 문서에서 확인하세요.
"""

MODE_PROMPTS = {
    "기본 Q&A": """
답변 방식:
- 사용자의 질문에 직접적으로 답변하세요.
- 불필요하게 길게 설명하지 말고 핵심을 중심으로 답변하세요.
""",
    "개념 설명": """
답변 방식:
- 개념을 처음 배우는 학생도 이해할 수 있도록 설명하세요.
- 정의, 핵심 원리, 예시 순서로 설명하세요.
- 필요한 경우 관련 용어도 함께 풀어서 설명하세요.
- 용어의 정식 명칭·약어는 참고 문서에 나온 표현만 사용하세요. 강의 제목, 슬라이드 제목, 파일명을 약어의 풀이로 쓰지 마세요.
- 예시는 참고 문서에 등장하는 설명, 흐름, 키워드만 활용하세요. 문서에 없는 연도, 사례(예: 특정 연도의 과학 발견)를 새로 만들지 마세요.
""",
    "시험 대비 요약": """
답변 방식:
- 시험 직전에 복습할 수 있도록 핵심 내용을 요약하세요.
- 중요한 키워드와 개념 위주로 정리하세요.
- 가능하면 번호 목록이나 소제목을 사용하세요.
""",
    "예상 문제 생성": """
답변 방식:
- 참고 문서를 바탕으로 예상 문제를 생성하세요.
- 객관식 또는 서술형 문제를 포함하세요.
- 각 문제 아래에 정답과 간단한 해설을 함께 작성하세요.
- 자료에 근거하지 않은 문제는 만들지 마세요.
""",
}


def _truncate_history_content(role: str, content: str, max_chars: int) -> str:
    """assistant 답변만 길이 제한 (토큰·비용 절감). user 질문은 전문 유지."""
    if role != "assistant" or len(content) <= max_chars:
        return content
    return content[: max_chars - 3].rstrip() + "..."


def format_chat_history_block(
    chat_history: Optional[List[Dict[str, Any]]] = None,
    max_turns: int = MAX_CHAT_HISTORY_TURNS,
    max_chars_per_message: int = CHAT_HISTORY_MAX_CHARS,
) -> str:
    """프롬프트에 넣을 이전 대화 텍스트."""
    if not chat_history:
        return "(이전 대화 없음)"

    role_label = {"user": "학습자", "assistant": "도우미"}
    lines: List[str] = []

    for msg in chat_history[-(max_turns * 2) :]:
        role = msg.get("role")
        if role not in role_label:
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        content = _truncate_history_content(role, content, max_chars_per_message)
        lines.append(f"{role_label[role]}: {content}")

    if not lines:
        return "(이전 대화 없음)"

    return "\n".join(lines)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2,
    )


def build_prompt(answer_mode: str = "기본 Q&A") -> ChatPromptTemplate:
    mode_prompt = MODE_PROMPTS.get(answer_mode, MODE_PROMPTS["기본 Q&A"])
    system_prompt = BASE_RULES + "\n" + mode_prompt

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                """
이전 대화:
{conversation}

현재 질문:
{question}

참고 문서:
{context}

위 참고 문서를 바탕으로 현재 질문에 답변하세요. 이전 대화는 맥락 이해용이며, 사실은 참고 문서만 사용하세요.
""",
            ),
        ]
    )
