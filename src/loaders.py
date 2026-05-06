from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(data_dir: str) -> List[Document]:
    """
    data_dir 폴더 안의 PDF/TXT 파일을 LangChain Document 객체로 로드합니다.
    """
    documents: List[Document] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")

    for file_path in data_path.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            continue

        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded_docs = loader.load()

        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            loaded_docs = loader.load()

        else:
            continue

        for doc in loaded_docs:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_path"] = str(file_path)

            # PDF는 PyPDFLoader가 page metadata를 넣어주는 경우가 많음
            # TXT는 page 개념이 없으므로 기본값 처리
            if "page" not in doc.metadata:
                doc.metadata["page"] = None

        documents.extend(loaded_docs)

    return documents