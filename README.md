# 강의자료 기반 RAG 학습 도우미

본 프로젝트는 강의자료를 기반으로 사용자의 질문에 답변하는 RAG 기반 학습 도우미입니다.

## 기술 스택

- Python
- Streamlit
- LangChain
- Chroma
- OpenAI Chat Model
- OpenAI Embedding

## 1차 구현 범위

- PDF/TXT 문서 로딩
- 문서 청킹
- OpenAI Embedding 생성
- Chroma Vector DB 저장
- 유사도 기반 문서 검색
- 검색 결과 기반 LLM 답변 생성
- 답변 출처 표시

## 향후 확장 계획

- PPTX/DOCX 문서 지원
- 사용자 파일 업로드 기능
- 검색 품질 평가
- 문서별 필터링
- 대화 기록 저장
- 관리자용 문서 관리 UI

## 실행 방법

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
`.env` 파일에 OpenAI API Key를 입력합니다.
```bash
OPENAI_API_KEY=your_openai_api_key_here
```
문서를 `data/raw/` 폴더에 넣은 뒤 실행합니다.
```bash
streamlit run app.py
```
