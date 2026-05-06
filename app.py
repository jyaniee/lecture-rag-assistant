import streamlit as st

from src.config import DATA_DIR
from src.loaders import load_documents
from src.splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_chain import ask_question


st.set_page_config(
    page_title="강의자료 기반 RAG 학습 도우미",
    page_icon="📚",
    layout="wide",
)


st.title("📚 강의자료 기반 RAG 학습 도우미")

st.caption(
    "강의자료를 기반으로 질문에 답변하고, 참고한 문서 출처를 함께 표시합니다."
)


with st.sidebar:
    st.header("설정")

    st.write("현재 1차 구현은 `data/raw` 폴더의 PDF/TXT 파일을 사용합니다.")
    st.code(DATA_DIR)

    if st.button("📦 문서 인덱싱하기"):
        with st.spinner("문서를 불러오고 벡터 DB를 생성하는 중입니다..."):
            documents = load_documents(DATA_DIR)

            if not documents:
                st.warning("불러올 문서가 없습니다. data/raw 폴더에 PDF 또는 TXT 파일을 넣어주세요.")
            else:
                chunks = split_documents(documents)
                create_vector_store(chunks)

                st.success(f"인덱싱 완료: 원본 문서 {len(documents)}개, 청크 {len(chunks)}개")


st.subheader("질문하기")

question = st.text_input(
    "강의자료에 대해 질문을 입력하세요.",
    placeholder="예: RAG의 핵심 과정은 무엇인가요?",
)

if st.button("질문하기", type="primary"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("답변을 생성하는 중입니다..."):
            try:
                result = ask_question(question)

                st.markdown("## 답변")
                st.write(result["answer"])

                st.markdown("## 참고한 자료")

                for i, source in enumerate(result["sources"], start=1):
                    page_text = f"p.{source['page']}" if source["page"] else "페이지 정보 없음"

                    with st.expander(f"{i}. {source['source']} / {page_text}"):
                        st.write(source["content"])

            except Exception as e:
                st.error("답변 생성 중 오류가 발생했습니다.")
                st.exception(e)