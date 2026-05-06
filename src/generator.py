from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_CHAT_MODEL


SYSTEM_PROMPT = """
당신은 강의자료 기반 학습 도우미입니다.

규칙:
1. 반드시 제공된 참고 문서 내용을 근거로 답변하세요.
2. 참고 문서에 없는 내용은 추측하지 말고 "제공된 자료만으로는 확인하기 어렵습니다"라고 답변하세요.
3. 답변은 한국어로 작성하세요.
4. 시험 답안처럼 개념을 명확하고 구조적으로 설명하세요.
5. 가능한 경우 답변 마지막에 참고한 문서명을 자연스럽게 언급하세요.
"""


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        temperature=0.2,
    )


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                """
질문:
{question}

참고 문서:
{context}

위 참고 문서를 바탕으로 답변하세요.
""",
            ),
        ]
    )