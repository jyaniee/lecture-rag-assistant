import re
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document

from src.generator import get_llm, build_prompt
from src.retriever import get_retriever, search_documents_with_score

RELEVANCE_SCORE_THRESHOLD = 1.2

def is_low_relevance(search_results, threshold: float = RELEVANCE_SCORE_THRESHOLD) -> bool:
    """검색 결과의 관련도가 낮은지 판단합니다."""
    if not search_results:
        return True
    
    best_score = search_results[0][1]

    return best_score > threshold

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


def ask_question(question: str, answer_mode: str = "기본 Q&A") -> Dict[str, Any]:
    """
    질문을 받아 RAG 방식으로 답변과 출처를 반환합니다.
    """
    search_results = search_documents_with_score(question, k=4)

    debug_scores = [
        {
            "file_name": doc.metadata.get("file_name", doc.metadata.get("source", "알 수 없는 문서")),
            "page": doc.metadata.get("page"),
            "score": score,
        }
        for doc, score in search_results
    ]

    if is_low_relevance(search_results):
        return {
            "answer": "제공된 강의자료에서 질문과 관련된 내용을 찾기 어렵습니다.",
            "sources": [],
            "answer_mode": answer_mode,
            "is_rejected": True,
            "debug_scores": debug_scores,
        }
    # retriever = get_retriever()
    # docs = retriever.invoke(question)
    docs = [doc for doc, _score in search_results]

    context = format_documents(docs)

    prompt = build_prompt(answer_mode)
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
        "answer_mode": answer_mode,
        "is_rejected": False,
        "debug_scores": debug_scores,
    }