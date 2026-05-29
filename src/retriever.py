from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from src.config import RETRIEVER_TOP_K
from src.vector_store import load_vector_store

from typing import List, Tuple



def get_retriever() -> VectorStoreRetriever:
    """
    Chroma DB에서 질문과 유사한 문서 청크를 검색하는 Retriever를 생성합니다.
    """
    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_TOP_K},
    )

def search_documents_with_score(
        query: str,
        k: int = RETRIEVER_TOP_K,
) -> List[Tuple[Document, float]]:
    """질문과 유사한 문서 청크를 점수와 함께 검색합니다."""
    vector_store = load_vector_store()

    return vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )