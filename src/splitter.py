from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENABLE_PARENT_CHILD,
    PARENT_CONTENT_MAX_CHARS,
)


def _parent_id(doc: Document) -> str:
    meta = doc.metadata or {}
    file_path = meta.get("file_path") or meta.get("source") or "unknown"
    page = meta.get("page")
    return f"{file_path}|{page}"


def _attach_parent_metadata(children: List[Document], parent_text: str, parent_id: str) -> List[Document]:
    capped = parent_text[:PARENT_CONTENT_MAX_CHARS]
    for child in children:
        child.metadata = dict(child.metadata or {})
        child.metadata["parent_id"] = parent_id
        child.metadata["parent_content"] = capped
        child.metadata["is_child_chunk"] = True
    return children


def split_documents(documents: List[Document]) -> List[Document]:
    """
    문서를 RAG 검색용 청크로 분할합니다.

    ENABLE_PARENT_CHILD=true 이면 페이지/슬라이드 단위 parent를 유지하고
    작은 child 청크로 검색합니다 (재인덱싱 필요).
    """
    if not ENABLE_PARENT_CHILD:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks: List[Document] = []
    for doc in documents:
        parent_text = (doc.page_content or "").strip()
        if not parent_text:
            continue

        parent_id = _parent_id(doc)
        children = child_splitter.split_documents([doc])
        if len(children) == 1 and children[0].page_content == parent_text:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["parent_id"] = parent_id
            doc.metadata["parent_content"] = parent_text[:PARENT_CONTENT_MAX_CHARS]
            doc.metadata["is_child_chunk"] = False
            all_chunks.append(doc)
            continue

        all_chunks.extend(_attach_parent_metadata(children, parent_text, parent_id))

    return all_chunks
