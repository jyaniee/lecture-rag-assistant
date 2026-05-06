from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, OPENAI_EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)


def create_vector_store(documents: List[Document]) -> Chroma:
    """
    문서 청크를 임베딩한 뒤 Chroma DB에 저장합니다.
    기존 DB가 있으면 덮어쓰기보다는 같은 경로에 추가될 수 있으므로,
    필요하면 vectorstore/chroma 폴더를 삭제한 뒤 재생성하세요.
    """
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    이미 생성된 Chroma DB를 불러옵니다.
    """
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )