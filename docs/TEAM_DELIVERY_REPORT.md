# 팀 전달 보고 — RAG 학습 도우미 (dev-hm)


---

## 한 줄 요약

강의자료 RAG의 **질문 → 검색 → 답변 → 출처** 흐름을 안정화했습니다. **UI는 Streamlit(`app.py`) 단일**이며, 별도 프론트엔드는 없습니다.  
**과목(subject) 필수 선택**, **질의 전처리·Kiwi BM25**, **하이브리드 검색·품질 실험**까지 한 파이프라인으로 묶었습니다.

**이번 PR/커밋에 권장 메시지:**

```text
feat: RAG 고도화 — 과목 필수·전처리·Kiwi·하이브리드·품질실험

- Streamlit 단일 UI, 사이드바 과목 선택·index_data_dir 인덱싱 통일
- 질의 전처리·Kiwi BM25, subject 없으면 거절
- BM25+벡터 RRF, rerank, parent-child, 품질 실험 --subject 필수
```

---

## 왜 이렇게까지 했는지

프로젝트 성격이 **「강의 RAG 학습 도우미」**라서, UI보다 먼저 아래가 맞아야 한다고 봤습니다.

1. **검색이 틀리면** 답도 틀리고, 토큰만 낭비됩니다.  
2. **환각·무관 답변·과목 혼선**은 강의 도우미에서 치명적입니다.  
3. 품질을 말할 **근거**(`retrieval` 메타, `quality_experiments`)가 필요합니다.

README 「향후」는 팀 정책상 **수정하지 않았습니다.** 이 보고서와 코드·`.env.example`이 실제 구현 기준입니다.

---

## UI·운영 방식 (Streamlit 단일)

| 항목 | 내용 |
|------|------|
| **UI** | `streamlit run app.py` 만 사용. 별도 React/Vue 등 프론트 없음 |
| **과목 선택** | 사이드바 selectbox → `ask_question(..., subject=선택과목)` |
| **과목 미선택·빈 값** | `rag_chain`이 `subject_required`로 거절 (LLM 미호출) |
| **인덱싱** | 사이드바 「문서 인덱싱하기」→ `index_data_dir(DATA_DIR, reset=...)` (로직 통일) |
| **지원 형식** | PDF, TXT, DOCX, PPTX (`loaders.SUPPORTED_EXTENSIONS`) |
| **데이터 배치** | `data/raw/{과목폴더명}/` → 인덱싱 시 `metadata.subject` = 폴더명 |

---

## 이번에 넣는 작업 — 의미 / 목적 / 기대 효과

### 1) 문서 로딩·메타 (`loaders.py`)

- **의미:** PDF/TXT/DOCX/PPTX 지원, 모든 청크에 `subject`, `file_name`, `file_path` 등 메타 일관 적용.  
- **목적:** 과목·파일 필터·출처 표시의 기반.  
- **기대 효과:** 재인덱싱 후 검색·출처가 파일/과목 단위로 맞음.  
- **팀 주의:** 로더·형식 변경 후 **재인덱싱** 필요.

### 2) 질의 전처리·한글 토큰화 (신규)

| 모듈 | 역할 |
|------|------|
| `query_preprocess.py` | 라틴 약어 대문자화, 한글 띄어쓰기 규칙, 모호 질문·thin context 가드 |
| `ko_tokenizer.py` | Kiwi(`kiwipiepy`) BM25 토큰화, 실패 시 정규식 폴백 |

- **목적:** 한글 질의·키워드 검색 recall 개선, 무의미·짧은 질문 조기 차단.  
- **연동:** `hybrid_search.py` BM25, `rag_chain` 검색 전 전처리.

### 3) 검색·답변 코어 (`retriever.py`, `rag_chain.py`, `generator.py`)

- **의미:** 유사도 검색 → **관련도 임계값** → 파일당 1청크 **dedup** → LLM. 무관하면 **LLM 호출 없이** 거부.  
- **과목 필터:** `subject` 없으면 거절; 있으면 Chroma `where`로 해당 과목만 검색.  
- **기대 효과:** 「날씨」·타 과목 질문은 빠르게 거부, 선택 과목 강의 질문은 근거 있는 답·출처.  
- **추가:** `result["retrieval"]`로 `best_score`, `context_chunk_count`, `rejection_reason` 등 **품질 설명 가능**.

### 4) 벡터 DB·운영 API (`vector_store.py`)

- **Streamlit이 쓰는 API:** `index_data_dir`, `list_indexed_subjects`, (내부) `create_vector_store` / `reset_vector_store`  
- **예비 API (Streamlit 미사용, 코드 유지):** 아래 표 참고. 향후 관리 화면·업로드 UI 연동용.

| 함수 / 설정 | 용도 | Streamlit |
|-------------|------|-----------|
| `ingest_uploaded_files` | 업로드 파일 인덱싱 | 미사용 |
| `list_indexed_documents` | 인덱스 문서 목록 | 미사용 |
| `delete_indexed_document` | 문서 삭제 | 미사용 |
| `get_index_stats` | 통계 | 미사용 |
| `get_retriever()` (`retriever.py`) | LangChain retriever 직접 노출 | 미사용 |
| `UPLOAD_DIR` (`config.py`) | 업로드 저장 경로 | 미사용 |

- Windows **WinError 32** 시 컬렉션 삭제 우선 (재인덱싱 전 Streamlit 종료).

### 5) 다턴 대화 + 후속 질문 검색 (`rag_chain`, `app.py`)

- **의미:** 매 턴 **다시 Chroma 검색**. `chat_history`는 맥락·짧은 후속 질문용 **query 보강**.  
- **목적:** 「방금 말한 Retrieval만 더」 같은 질문에서 검색 실패 줄이기.  
- **기대 효과:** 세션 안에서 이어 말하기 가능. **영구 저장**은 미구현.

### 6) 품질 실험·튜닝 (`quality_experiments.py`)

- **의미:** 유형별 질문 일괄 실행, **`--probe-only`**(LLM 없음), **threshold/k 스윕**, JSON 저장.  
- **`--subject` 필수** — 자동 과목 선택 없음 (과목 혼선 방지).  
- **산출물:** `reports/` (로컬·튜닝용, `.gitignore` 제외)  
- **담당자 실험 예:**

```bash
python -m src.quality_experiments --probe-only --subject 고급인공지능
python -m src.quality_experiments --sweep-threshold --subject 고급인공지능
```

### 7) 고급 RAG — 모듈

| 기능 | 파일 | 목적 | 기대 효과 |
|------|------|------|-----------|
| Chroma·임베딩 **싱글톤** | `vector_store.py` | 매 요청 객체 재생성 제거 | 지연·안정성 |
| **BM25 + 벡터 RRF** | `hybrid_search.py` | 키워드·의미 둘 다 (Kiwi 토큰) | 용어·약어 질문 recall ↑ |
| **겹침 rerank** | `rerank.py` | fetch 넓힌 뒤 재정렬 | 관련 청크가 위로 |
| **extractive 압축** | `context_compress.py` | LLM에 넣는 길이 제한 | 토큰·비용 ↓ |
| **parent–child** | `splitter.py` | 작은 청크로 찾고 parent로 읽기 | PPTX·긴 PDF 맥락 ↑ |
| **스트리밍** | `rag_chain.ask_question_events` | 토큰 단위 이벤트 | `app.py` 스트리밍 데모 |

- **env:** `.env.example`에 기능 on/off·상수 정리.  
- **의존성:** `rank-bm25`, `kiwipiepy` (`requirements.txt`).

**중요:** parent–child·BM25는 **재인덱싱 후**에야 인덱스와 맞습니다.

---

## Streamlit에서 아직 안 하는 것 (의도적)

| 기능 | 상태 | 비고 |
|------|------|------|
| 파일 업로드 UI | 미구현 | `ingest_uploaded_files` 예비 |
| 문서 목록·삭제 UI | 미구현 | `list_*`, `delete_*` 예비 |
| `retrieval` 디버그 패널 | 부분 | 답변·출처는 표시, 상세 메타는 확장 여지 |
| 별도 웹 프론트 | 없음 | Streamlit만 사용 |

---

## pull 받은 뒤 할 일 (체크리스트)

1. `.venv` 활성화 후 `pip install -r requirements.txt` (`rank-bm25`, `kiwipiepy` 포함)  
2. `copy .env.example .env` 후 키 입력  
3. `streamlit run app.py`  
4. 사이드바 **과목 선택** 후 질문 (과목 필수)  
5. 사이드바 **문서 인덱싱** (가능하면 「벡터 DB 초기화」 후)  
6. `python -m src.quality_experiments --probe-only --subject <과목명>`  

**WinError 32** 나오면: Streamlit 끄고 다시 인덱싱.

---

## 점검해 둔 것 / 남은 리스크

**해 둔 것**

- `src` 문법·린트  
- `subject=고급인공지능` RAG 질문 스모크  
- probe 품질 실험: **pass 4 / fail 0 / manual 2** (해당 과목·threshold 기준)  
- Streamlit 인덱싱 → `index_data_dir` 통일  

**남은 것 (버그라기보다 운영·범위)**

- **재인덱싱 전** parent–child·BM25 효과 제한적  
- full LLM 품질 실험·수동 평가 → 담당자 실행  
- 대용량 코퍼스 시 BM25 메모리 — 현재 강의 규모 가정  
- 예비 API 코드 삭제 — P2에서 팀 합의 후  

---

## 파일 변경 요약 (리뷰용)

| 영역 | 주요 파일 |
|------|-----------|
| 신규 | `query_preprocess.py`, `ko_tokenizer.py`, `quality_experiments.py`, `hybrid_search.py`, `rerank.py`, `context_compress.py`, `.env.example` |
| 수정 | `app.py`, `loaders.py`, `vector_store.py`, `rag_chain.py`, `retriever.py`, `splitter.py`, `config.py`, `generator.py`, `requirements.txt`, `.gitignore` |
| Git | `docs/TEAM_DELIVERY_REPORT.md` 만 추적 (`reports/`·`docs/*` 는 ignore) |

---

## 문의 시 참고 API

**Streamlit·일상 사용**

```python
from src.rag_chain import ask_question, ask_question_events, run_retrieval
from src.vector_store import index_data_dir, list_indexed_subjects
```

- 답변: `ask_question(질문, 모드, chat_history=..., subject=필수, file_name=...)`  
- 스트리밍: `for e in ask_question_events(...):` → `token` / `final`  
- 검색만: `run_retrieval(질문, subject=...)`  

**예비 (Streamlit 미연동)**

```python
from src.vector_store import ingest_uploaded_files, list_indexed_documents, delete_indexed_document, get_index_stats
from src.retriever import get_retriever
```

---

README와 다른 부분은 **이 보고서·코드·`.env.example`** 을 우선해 주시면 됩니다.
