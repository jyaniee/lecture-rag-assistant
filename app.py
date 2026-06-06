from pathlib import Path
import html

import streamlit as st

from src.config import DATA_DIR, ENABLE_LLM_STREAMING, MAX_CHAT_HISTORY_TURNS
from src.loaders import SUPPORTED_EXTENSIONS, load_documents
from src.splitter import split_documents
from src.vector_store import index_data_dir, list_indexed_subjects
from src.rag_chain import ask_question, ask_question_events


st.set_page_config(
    page_title="강의자료 기반 RAG 학습 도우미",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(88, 28, 135, 0.32), transparent 42%),
            radial-gradient(circle at top right, rgba(30, 64, 175, 0.18), transparent 36%),
            linear-gradient(180deg, #05070D 0%, #080B12 50%, #0B0F19 100%);
        color: #E5E7EB;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }

    header[data-testid="stHeader"] > div {
        background: transparent;
    }

    div[data-testid="stToolbar"] {
        background: transparent;
    }

    div[data-testid="stToolbar"] * {
        background: transparent;
    }

    div[data-testid="stDecoration"] {
        background: transparent;
    }
    .block-container {
        max-width: 980px;
        padding-top: 2.2rem;
        padding-bottom: 6rem;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
    }

    .main-subtitle {
        font-size: 1rem;
        color: #A0AEC0;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }

    .hero-card {
        padding: 1.15rem 1.35rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.22),
            rgba(37, 99, 235, 0.12)
        );
        border: 1px solid rgba(167, 139, 250, 0.28);
        margin-bottom: 1.5rem;
    }

    .hero-card-title {
        font-size: 1.1rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .hero-card-desc {
        color: #CBD5E1;
        line-height: 1.65;
        font-size: 0.95rem;
    }

    .empty-chat-card {
        padding: 1.2rem 1.35rem;
        border-radius: 18px;
        background-color: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.2);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .empty-chat-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .empty-chat-desc {
        color: #A0AEC0;
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.9rem;
    }

    .suggestion-card {
        padding: 0.8rem 0.9rem;
        border-radius: 14px;
        background-color: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #CBD5E1;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .mode-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background-color: rgba(167, 139, 250, 0.16);
        color: #C4B5FD;
        border: 1px solid rgba(167, 139, 250, 0.28);
        font-size: 0.82rem;
        margin-bottom: 0.7rem;
    }

    .source-header {
        font-size: 0.95rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.55rem;
    }

    .source-card {
        padding: 0.8rem 0.95rem;
        border-radius: 14px;
        background-color: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.22);
        margin-bottom: 0.55rem;
    }

    .source-title {
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #E5E7EB;
        font-size: 0.92rem;
    }

    .source-meta {
        color: #94A3B8;
        font-size: 0.84rem;
    }

    .small-muted {
        color: #94A3B8;
        font-size: 0.84rem;
    }

    /* 사이드바 전체 */
    div[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(3, 7, 18, 0.96) 0%, rgba(8, 13, 24, 0.94) 48%, rgba(3, 7, 18, 0.98) 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.14);
    }

    /* 사이드바 실제 콘텐츠 영역 */
    div[data-testid="stSidebarContent"] {
        background:
            radial-gradient(circle at top left, rgba(88, 28, 135, 0.20), transparent 38%),
            linear-gradient(180deg, rgba(15, 23, 42, 0.42), rgba(3, 7, 18, 0.32));
        padding: 1.4rem 1.05rem;
    }

    /* 사이드바 제목 */
    div[data-testid="stSidebar"] h1 {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #F8FAFC;
        margin-bottom: 1rem;
    }

    /* 사이드바 섹션 제목 */
    div[data-testid="stSidebar"] h3 {
        font-size: 0.95rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #DDD6FE;
        margin-top: 1.15rem;
        margin-bottom: 0.55rem;
    }

    /* 사이드바 일반 텍스트 */
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span,
    div[data-testid="stSidebar"] label {
        font-size: 0.82rem;
        line-height: 1.4;
        color: #CBD5E1;
    }

    /* 사이드바 caption */
    div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #94A3B8;
    }

    /* 사이드바 버튼 */
    div[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.72), rgba(30, 64, 175, 0.48));
        border: 1px solid rgba(167, 139, 250, 0.28);
        color: #F8FAFC;
        font-weight: 700;
        transition: 0.15s ease;
    }

    div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
        border-color: rgba(196, 181, 253, 0.58);
        background: linear-gradient(135deg, rgba(109, 40, 217, 0.82), rgba(37, 99, 235, 0.58));
        transform: translateY(-1px);
    }

    /* 사이드바 selectbox */
    div[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        color: #E5E7EB;
    }

    div[data-testid="stSidebar"] div[data-baseweb="select"] > div:hover {
        border-color: rgba(167, 139, 250, 0.36);
    }

    /* 사이드바 expander */
    div[data-testid="stSidebar"] div[data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.10);
        background-color: rgba(15, 23, 42, 0.34);
        margin-bottom: 0.45rem;
    }

    div[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        color: #CBD5E1;
        font-weight: 650;
        font-size: 0.86rem;
    }

    /* 사이드바 파일 목록 */
    .file-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.34rem 0.45rem;
        border-radius: 8px;
        color: #CBD5E1;
        font-size: 0.82rem;
        line-height: 1.35;
        margin-bottom: 0.15rem;
    }

    .file-item:hover {
        background-color: rgba(148, 163, 184, 0.10);
    }

    .file-icon {
        opacity: 0.8;
        font-size: 0.82rem;
        flex-shrink: 0;
    }

    .file-name {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* 구분선 */
    div[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.14);
        margin: 1.15rem 0;
    }

    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 650;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 0.4rem 0;
    }

    div[data-testid="stChatInput"] textarea {
        border-radius: 16px;
        min-height: 48px;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px;
        border-color: rgba(148, 163, 184, 0.22);
    }

    @media (max-width: 760px) {
        .suggestion-grid {
            grid-template-columns: 1fr;
        }

        .main-title {
            font-size: 1.8rem;
        }
    }
    /* 사용자 메시지 우측 정렬 */
    .user-message-row {
        display: flex;
        justify-content: flex-end;
        margin: 0.75rem 0;
    }

    .user-message-bubble {
        max-width: 72%;
        padding: 0.85rem 1rem;
        border-radius: 18px 18px 4px 18px;
        background:
            linear-gradient(
                135deg,
                rgba(124, 58, 237, 0.34),
                rgba(37, 99, 235, 0.20)
            ),
            rgba(15, 23, 42, 0.84);
        border: 1px solid rgba(167, 139, 250, 0.28);
        color: #F8FAFC;
        line-height: 1.6;
        font-size: 0.95rem;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_available_files() -> list[Path]:
    data_path = Path(DATA_DIR)
    files = []

    if not data_path.exists():
        return files

    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)

    return sorted(files)


def get_available_subjects() -> list[str]:
    """인덱스된 과목 우선, 없으면 data/raw 폴더 구조에서 추출."""
    indexed = list_indexed_subjects()
    if indexed:
        return sorted(indexed)

    grouped = group_files_by_subject(get_available_files())
    return sorted(grouped.keys())


def _clear_chat_on_subject_change() -> None:
    st.session_state["chat_history"] = []


def group_files_by_subject(files: list[Path]) -> dict[str, list[Path]]:
    """파일 목록을 과목 폴더 기준으로 그룹화한다."""
    data_path = Path(DATA_DIR)
    grouped_files = {}

    for file_path in files:
        relative_path = file_path.relative_to(data_path)

        if len(relative_path.parts) >= 2:
            subject = relative_path.parts[0]
        else:
            subject = "기타"

        grouped_files.setdefault(subject, []).append(file_path)

    return grouped_files


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("표시할 참고 자료가 없습니다.")
        return

    grouped_sources = {}

    for source in sources:
        subject = source.get("subject", "과목 정보 없음")
        file_name = source.get("file_name", source.get("source", "알 수 없는 문서"))
        page = source.get("page")

        key = (subject, file_name)

        if key not in grouped_sources:
            grouped_sources[key] = {
                "subject": subject,
                "file_name": file_name,
                "pages": set(),
            }

        if page:
            grouped_sources[key]["pages"].add(page)

    st.markdown('<div class="source-header">참고한 자료</div>', unsafe_allow_html=True)

    for i, source in enumerate(grouped_sources.values(), start=1):
        subject = html.escape(source["subject"])
        file_name = html.escape(source["file_name"])

        pages = sorted(source["pages"])

        if pages:
            page_text = ", ".join(f"p.{page}" for page in pages)
        else:
            page_text = "페이지 정보 없음"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{i}. {file_name}</div>
                <div class="source-meta">과목: {subject} · 페이지: {page_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_debug_scores(debug_scores: list[dict]) -> None:
    with st.expander("검색 유사도 점수"):
        st.caption("점수가 낮을수록 질문과 문서의 관련도가 높습니다.")
        st.write(debug_scores)

def render_user_message(content: str) -> None:
    st.markdown(
        f"""
        <div class="user-message-row">
            <div class="user-message-bubble">
                {html.escape(content)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_assistant_message(message: dict, *, skip_body: bool = False) -> None:
    st.markdown(
        f'<div class="mode-badge">답변 모드: {html.escape(message.get("answer_mode", "기본 Q&A"))}</div>',
        unsafe_allow_html=True,
    )

    if not skip_body:
        if message.get("is_rejected"):
            st.warning(message["content"])
        else:
            st.write(message["content"])

    render_debug_scores(message.get("debug_scores", []))

    if message.get("sources"):
        render_sources(message["sources"])
    elif message.get("is_rejected"):
        st.info("표시할 참고 자료가 없습니다.")


def _prior_chat_history_for_rag(
    chat_history: list[dict],
    current_question: str,
) -> list[dict]:
    """현재 질문을 제외한 이전 user/assistant 메시지."""
    prior: list[dict] = []

    for message in chat_history:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (message.get("content") or "").strip()
        if not content:
            continue
        prior.append({"role": role, "content": content})

    if (
        prior
        and prior[-1]["role"] == "user"
        and prior[-1]["content"].strip() == current_question.strip()
    ):
        prior = prior[:-1]

    return prior[-(MAX_CHAT_HISTORY_TURNS * 2) :]


def render_empty_chat() -> None:
    st.markdown(
        """
        <div class="empty-chat-card">
            <div class="empty-chat-title">강의자료에 대해 질문해보세요.</div>
            <div class="empty-chat-desc">
                인덱싱된 강의자료를 기반으로 답변합니다.
                개념 설명, 시험 대비 요약, 예상 문제 생성처럼 학습 목적에 맞게 사용할 수 있습니다.
            </div>
            <div class="suggestion-grid">
                <div class="suggestion-card">LLM의 개념을 설명해줘.</div>
                <div class="suggestion-card">RAG가 환각을 줄이는 원리를 설명해줘.</div>
                <div class="suggestion-card">벡터 DB가 필요한 이유를 설명해줘.</div>
                <div class="suggestion-card">문서 기반 RAG 챗봇 구축 과정을 요약해줘.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

with st.sidebar:
    st.header("설정")

    st.markdown("### 자료 경로")
    st.code(DATA_DIR)

    st.markdown("---")
    st.markdown("### 인식된 자료")
    files = get_available_files()
    data_path = Path(DATA_DIR)

    if files:
        grouped_files = group_files_by_subject(files)

        st.caption(f"{len(files)}개 문서 인식됨")

        for subject, subject_files in grouped_files.items():
            with st.expander(f"{subject} ({len(subject_files)})", expanded=False):
                for file_path in subject_files:
                    relative_path = file_path.relative_to(data_path)
                    file_name = file_path.name

                    st.markdown(
                        f"""
                        <div class="file-item">
                            <span class="file-name">{html.escape(file_name)}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.caption("아직 인식 가능한 강의자료(PDF/TXT/DOCX/PPTX)가 없습니다.")

    st.markdown("---")
    st.markdown("### 인덱싱")

    reset_db = st.checkbox(
        "기존 벡터 DB 초기화 후 재인덱싱",
        value=True,
        help="기존 Chroma DB를 삭제하고 새로 인덱싱합니다.",
    )

    if st.button("문서 인덱싱하기", use_container_width=True):
        with st.spinner("문서를 불러오고 벡터 DB를 생성하는 중입니다..."):
            result = index_data_dir(DATA_DIR, reset=reset_db)

            if not result["success"]:
                err = (
                    result["errors"][0].get("message", "인덱싱에 실패했습니다.")
                    if result["errors"]
                    else "인덱싱에 실패했습니다."
                )
                st.warning(err)
            else:
                documents = load_documents(DATA_DIR)
                chunks = split_documents(documents)
                st.session_state["chunks_preview"] = chunks[:5]
                st.session_state["chunks_count"] = result["chunk_count"]
                st.success(
                    f"인덱싱 완료: 원본 문서 {result['document_count']}개, "
                    f"청크 {result['chunk_count']}개"
                )

    if "chunks_count" in st.session_state:
        st.caption(f"최근 생성된 청크 수: {st.session_state['chunks_count']}개")

    with st.expander("청크 미리보기", expanded=False):
        if "chunks_preview" in st.session_state:
            for i, chunk in enumerate(st.session_state["chunks_preview"], start=1):
                st.markdown(f"**청크 {i}**")
                st.write(chunk.page_content)
                st.caption(chunk.metadata)
                st.markdown("---")
        else:
            st.caption("인덱싱 후 청크 미리보기가 표시됩니다.")

    st.markdown("---")
    st.markdown("### 학습 과목")

    available_subjects = get_available_subjects()
    if available_subjects:
        if "selected_subject" not in st.session_state:
            st.session_state["selected_subject"] = available_subjects[0]

        current = st.session_state.get("selected_subject")
        if current not in available_subjects:
            st.session_state["selected_subject"] = available_subjects[0]

        st.selectbox(
            "검색·답변에 사용할 과목",
            available_subjects,
            key="selected_subject",
            on_change=_clear_chat_on_subject_change,
            help="선택한 과목 자료만 검색합니다. 과목을 바꾸면 대화가 초기화됩니다.",
        )
    else:
        st.session_state["selected_subject"] = None
        st.caption("인덱싱된 과목이 없습니다. 자료를 넣고 인덱싱해 주세요.")

    st.markdown("---")
    st.markdown("### 답변 설정")

    answer_mode = st.selectbox(
        "답변 모드",
        ["기본 Q&A", "개념 설명", "시험 대비 요약", "예상 문제 생성"],
    )

    st.markdown("---")
    st.markdown("### 대화")

    if st.button("대화 초기화", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


st.markdown(
    """
    <div class="main-title">강의자료 기반 RAG 학습 도우미</div>
    <div class="main-subtitle">
        강의자료를 검색해 근거 기반 답변, 요약, 예상 문제 생성을 지원합니다.
    </div>
    """,
    unsafe_allow_html=True,
)

selected_subject = st.session_state.get("selected_subject")
subject_label = html.escape(selected_subject) if selected_subject else "미선택"

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-card-title">자료에 근거한 학습형 AI Assistant</div>
        <div class="hero-card-desc">
            선택한 과목(<strong>{subject_label}</strong>) 자료만 검색해 답변합니다.
            검색 유사도가 낮은 질문은 답변을 제한하여 환각 가능성을 줄입니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not selected_subject:
    st.warning("사이드바에서 학습 과목을 선택하거나, 자료 인덱싱을 먼저 진행해 주세요.")


if not st.session_state["chat_history"]:
    render_empty_chat()


for message in st.session_state["chat_history"]:
    if message["role"] == "user":
        render_user_message(message["content"])
    else:
        with st.chat_message("assistant"):
            render_assistant_message(message)


prompt = st.chat_input(
    "강의자료에 대해 질문해보세요. 예: LLM의 개념을 설명해줘."
)

if prompt:
    if not selected_subject:
        st.error("과목이 선택되지 않아 질문을 처리할 수 없습니다. 사이드바에서 과목을 선택해 주세요.")
    else:
        user_message = {
            "role": "user",
            "content": prompt,
        }

        st.session_state["chat_history"].append(user_message)

        render_user_message(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                try:
                    prior_history = _prior_chat_history_for_rag(
                        st.session_state["chat_history"],
                        prompt,
                    )

                    stream_placeholder = st.empty()
                    streamed_answer = False

                    if ENABLE_LLM_STREAMING:
                        result = None
                        stream_parts: list[str] = []
                        for event in ask_question_events(
                            prompt,
                            answer_mode,
                            subject=selected_subject,
                            chat_history=prior_history,
                        ):
                            if event["type"] == "token":
                                stream_parts.append(event["content"])
                                stream_placeholder.markdown("".join(stream_parts))
                            elif event["type"] == "final":
                                result = event["data"]
                        if result is None:
                            raise RuntimeError("스트리밍 응답이 완료되지 않았습니다.")
                        streamed_answer = bool(stream_parts) and not result.get(
                            "is_rejected", False
                        )
                    else:
                        result = ask_question(
                            prompt,
                            answer_mode,
                            subject=selected_subject,
                            chat_history=prior_history,
                        )

                    assistant_message = {
                        "role": "assistant",
                        "content": result["answer"],
                        "answer_mode": result.get("answer_mode", answer_mode),
                        "sources": result.get("sources", []),
                        "debug_scores": result.get("debug_scores", []),
                        "is_rejected": result.get("is_rejected", False),
                    }

                    if streamed_answer:
                        stream_placeholder.markdown(result["answer"])
                        assistant_message["content"] = result["answer"]
                        render_assistant_message(assistant_message, skip_body=True)
                    else:
                        stream_placeholder.empty()
                        render_assistant_message(assistant_message)

                    st.session_state["chat_history"].append(assistant_message)

                except Exception as e:
                    error_message = "답변 생성 중 오류가 발생했습니다."

                    assistant_message = {
                        "role": "assistant",
                        "content": error_message,
                        "answer_mode": answer_mode,
                        "sources": [],
                        "debug_scores": [],
                        "is_rejected": True,
                    }

                    st.error(error_message)
                    st.exception(e)

                    st.session_state["chat_history"].append(assistant_message)