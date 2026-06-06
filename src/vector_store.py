from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, DATA_DIR, OPENAI_EMBEDDING_MODEL
from src.loaders import load_documents, load_documents_from_paths
from src.splitter import split_documents


PathLike = Union[str, Path]

_embeddings_singleton: Optional[OpenAIEmbeddings] = None
_chroma_singleton: Optional[Chroma] = None
_chunks_cache: Optional[List[Document]] = None


def invalidate_search_caches() -> None:
    """인덱스 변경·재로딩 시 검색 캐시를 비웁니다."""
    global _chunks_cache
    _chunks_cache = None
    try:
        from src.hybrid_search import clear_bm25_cache

        clear_bm25_cache()
    except ImportError:
        pass


def invalidate_vector_store_cache() -> None:
    """Chroma·청크 캐시를 비웁니다 (컬렉션 삭제·재생성 후 호출)."""
    global _chroma_singleton
    _chroma_singleton = None
    invalidate_search_caches()


def get_embedding_model() -> OpenAIEmbeddings:
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    return _embeddings_singleton


def _clear_chroma_collection() -> bool:
    """Chroma 컬렉션의 모든 청크를 삭제합니다. Streamlit 등에서 DB 파일이 열려 있어도 동작합니다."""
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        return True

    try:
        store = load_vector_store()
        data = store._collection.get(include=[])
        ids = data.get("ids") or []
        if ids:
            store._collection.delete(ids=ids)
        invalidate_vector_store_cache()
        return True
    except Exception:
        return False


def reset_vector_store() -> None:
    """
    벡터 DB를 비웁니다.

    Windows(Streamlit)에서는 디렉터리 삭제(rmtree) 시 WinError 32가 날 수 있어,
    우선 컬렉션 내 청크 전체 삭제를 시도합니다.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        invalidate_vector_store_cache()
        return

    if _clear_chroma_collection():
        return

    try:
        shutil.rmtree(chroma_path)
        invalidate_vector_store_cache()
    except PermissionError as e:
        raise PermissionError(
            "Chroma DB 파일이 다른 프로세스에서 사용 중입니다. "
            "Streamlit을 재시작한 뒤 다시 인덱싱하거나, "
            "「벡터 DB 초기화」 체크를 해제하고 추가만 시도하세요."
        ) from e


def create_vector_store(documents: List[Document]) -> Chroma:
    invalidate_vector_store_cache()
    embeddings = get_embedding_model()
    store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    global _chroma_singleton
    _chroma_singleton = store
    return store


def load_vector_store() -> Chroma:
    global _chroma_singleton
    if _chroma_singleton is None:
        embeddings = get_embedding_model()
        _chroma_singleton = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
        )
    return _chroma_singleton


def _chroma_has_data() -> bool:
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        return False
    try:
        return load_vector_store()._collection.count() > 0
    except Exception:
        return False


def get_index_stats() -> Dict[str, Any]:
    """인덱스 요약 (프론트 상태 표시용)."""
    if not _chroma_has_data():
        return {"exists": Path(CHROMA_DIR).exists(), "chunk_count": 0}

    store = load_vector_store()
    return {"exists": True, "chunk_count": store._collection.count()}


def _matches_filter(meta: Dict[str, Any], subject: Optional[str], file_name: Optional[str]) -> bool:
    if subject and meta.get("subject") != subject:
        return False
    if file_name and meta.get("file_name") != file_name:
        return False
    return True


def get_indexed_chunks_for_search(
    subject: Optional[str] = None,
    file_name: Optional[str] = None,
) -> List[Document]:
    """BM25·하이브리드용 청크 목록 (메모리 캐시)."""
    global _chunks_cache

    if _chunks_cache is None:
        if not _chroma_has_data():
            return []

        store = load_vector_store()
        data = store._collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        _chunks_cache = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(documents, metadatas)
            if content
        ]

    if not subject and not file_name:
        return list(_chunks_cache)

    return [
        doc
        for doc in _chunks_cache
        if _matches_filter(doc.metadata or {}, subject, file_name)
    ]


def _add_chunks_to_store(chunks: List[Document]) -> int:
    if not chunks:
        return 0

    if _chroma_has_data():
        store = load_vector_store()
        ids = store.add_documents(chunks)
        invalidate_search_caches()
        return len(ids)

    store = create_vector_store(chunks)
    return store._collection.count()


def index_data_dir(
    data_dir: Optional[str] = None,
    *,
    reset: bool = True,
) -> Dict[str, Any]:
    """
    data_dir 아래 PDF/TXT/DOCX/PPTX를 로드·청킹·인덱싱합니다.
    app.py 「문서 인덱싱하기」+ reset 체크박스와 동일 역할.
    """
    target_dir = data_dir or DATA_DIR
    result: Dict[str, Any] = {
        "success": False,
        "document_count": 0,
        "chunk_count": 0,
        "indexed_files": [],
        "errors": [],
    }

    try:
        if reset:
            reset_vector_store()

        documents = load_documents(target_dir)
        if not documents:
            result["errors"].append({"message": "불러올 문서가 없습니다."})
            return result

        chunks = split_documents(documents)
        if reset:
            store = create_vector_store(chunks)
            chunk_count = store._collection.count()
        else:
            chunk_count = _add_chunks_to_store(chunks)

        file_names = sorted(
            {d.metadata.get("file_name") for d in documents if d.metadata.get("file_name")}
        )

        result["success"] = True
        result["document_count"] = len(documents)
        result["chunk_count"] = chunk_count
        result["indexed_files"] = [{"file_name": name} for name in file_names]
        return result

    except Exception as e:
        result["errors"].append({"message": str(e)})
        return result


def ingest_uploaded_files(
    file_paths: List[PathLike],
    *,
    reset: bool = False,
) -> Dict[str, Any]:
    """
    업로드된 파일 경로 목록만 인덱싱합니다.
    reset=False(기본): 기존 DB에 추가.
    """
    result: Dict[str, Any] = {
        "success": False,
        "document_count": 0,
        "chunk_count": 0,
        "indexed_files": [],
        "errors": [],
    }

    paths = [Path(p) for p in file_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        result["errors"].append({"message": "파일을 찾을 수 없습니다.", "paths": missing})
        return result

    try:
        if reset:
            reset_vector_store()

        documents = load_documents_from_paths(paths)
        if not documents:
            result["errors"].append({"message": "지원 형식(PDF/TXT/DOCX/PPTX) 문서가 없습니다."})
            return result

        chunks = split_documents(documents)
        if reset:
            store = create_vector_store(chunks)
            added = store._collection.count()
        else:
            added = _add_chunks_to_store(chunks)

        by_file: Dict[str, int] = {}
        for doc in documents:
            name = doc.metadata.get("file_name", "unknown")
            by_file[name] = by_file.get(name, 0) + 1

        result["success"] = True
        result["document_count"] = len(documents)
        result["chunk_count"] = added
        result["indexed_files"] = [
            {"file_name": k, "page_count": v} for k, v in sorted(by_file.items())
        ]
        return result

    except Exception as e:
        result["errors"].append({"message": str(e)})
        return result


def list_indexed_subjects() -> List[str]:
    """Chroma에 인덱싱된 고유 subject(과목) 목록."""
    if not _chroma_has_data():
        return []

    store = load_vector_store()
    data = store._collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    subjects: set[str] = set()
    for meta in metadatas:
        if meta and meta.get("subject"):
            subjects.add(str(meta["subject"]))

    return sorted(subjects)


def list_indexed_documents() -> List[Dict[str, Any]]:
    """Chroma에 인덱싱된 문서를 file_path 기준으로 묶어 반환합니다."""
    if not _chroma_has_data():
        return []

    store = load_vector_store()
    data = store._collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    grouped: Dict[str, Dict[str, Any]] = {}
    for meta in metadatas:
        if not meta:
            continue

        file_path = meta.get("file_path") or meta.get("source") or "unknown"
        key = str(file_path)

        if key not in grouped:
            grouped[key] = {
                "file_path": key,
                "file_name": meta.get("file_name") or Path(key).name,
                "subject": meta.get("subject"),
                "source": meta.get("source"),
                "chunk_count": 0,
            }
        grouped[key]["chunk_count"] += 1

    return sorted(
        grouped.values(),
        key=lambda x: (x.get("subject") or "", x.get("file_name") or ""),
    )


def delete_indexed_document(
    *,
    file_name: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """메타데이터 기준으로 청크를 삭제합니다."""
    result: Dict[str, Any] = {
        "success": False,
        "deleted_chunk_count": 0,
        "errors": [],
    }

    if not file_name and not file_path:
        result["errors"].append({"message": "file_name 또는 file_path가 필요합니다."})
        return result

    if not Path(CHROMA_DIR).exists():
        result["errors"].append({"message": "인덱스가 없습니다."})
        return result

    try:
        store = load_vector_store()
        before = store._collection.count()

        if file_path:
            where = {"file_path": file_path}
        else:
            where = {"file_name": file_name}

        store._collection.delete(where=where)
        after = store._collection.count()
        deleted = max(before - after, 0)
        invalidate_search_caches()

        result["success"] = True
        result["deleted_chunk_count"] = deleted
        return result

    except Exception as e:
        result["errors"].append({"message": str(e)})
        return result
