from typing import Dict, Any, List

from langchain_core.documents import Document

from src.generator import get_llm, build_prompt
from src.retriever import get_retriever


def format_documents(documents: List[Document]) -> str:
    """
    검색된 문서를 LLM 프롬프트에 넣기 좋은 문자열로 변환합니다.
    """
    formatted_chunks = []

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "알 수 없는 문서")
        page = doc.metadata.get("page", None)

        page_text = f"p.{page + 1}" if isinstance(page, int) else "페이지 정보 없음"

        formatted_chunks.append(
            f"[문서 {i}] 출처: {source}, {page_text}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_chunks)


def format_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    UI에서 출처를 표시하기 위한 구조화된 데이터로 변환합니다.
    """
    sources = []

    for doc in documents:
        source = doc.metadata.get("source", "알 수 없는 문서")
        page = doc.metadata.get("page", None)

        sources.append(
            {
                "source": source,
                "page": page + 1 if isinstance(page, int) else None,
                "content": doc.page_content[:500],
            }
        )

    return sources


def ask_question(question: str) -> Dict[str, Any]:
    """
    질문을 받아 RAG 방식으로 답변과 출처를 반환합니다.
    """
    retriever = get_retriever()
    docs = retriever.invoke(question)

    context = format_documents(docs)

    prompt = build_prompt()
    llm = get_llm()

    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )

    return {
        "answer": response.content,
        "sources": format_sources(docs),
    }