from typing import List
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, OPENAI_EMBEDDING_MODEL


def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    # OpenAI 임베딩 모델 생성

def reset_vector_store() -> None:
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        # 저장된 Chroma DB 디렉터리 삭제

def create_vector_store(documents: List[Document]) -> Chroma:   # 문서 청크 목록을 Chroma 벡터 저장소에 저장
    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return vector_store


def load_vector_store() -> Chroma:
    # 저장된 Chroma 벡터 저장소를 로드
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )