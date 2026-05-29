from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import OPENAI_CHAT_MODEL


BASE_RULES = """
당신은 강의자료 기반 학습 도우미입니다.

규칙:
1. 반드시 제공된 참고 문서 내용을 근거로 답변하세요.
2. 참고 문서에 없는 내용은 추측하지 말고 "제공된 자료만으로는 확인하기 어렵습니다"라고 답변하세요.
3. 답변은 한국어로 작성하세요.
4. 개념을 명확하고 구조적으로 설명하세요.
5. 가능한 경우 답변 마지막에 참고한 문서명을 자연스럽게 언급하세요.
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
""",
    "시험 대비 요약": """
답변 방식:
- 시험 직전에 복습할 수 있도록 핵심 내용을 요약하세요.
- 중요한 키워드와 개념 위주로 정리하세요.
- 가능하면 번호 목록이나 소제목을 사용하세요.
""",
    "예상 문제 생성": """
답변 상식:
- 참고 문서를 바탕으로 예상 문제를 생성하세요.
- 객관식 또는 서술형 문제를 포함하세요.
- 각 문제 아래에 정답과 간단한 해설을 함께 작성하세요.
- 자료에 근거하지 않은 문제는 만들지 마세요.
""",
}

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
질문:
{question}

참고 문서:
{context}

위 참고 문서를 바탕으로 답변하세요.
""",
            ),
        ]
    )