# 팀 전달 보고 — RAG 백엔드 (dev-hm)


---

## 한 줄 요약

강의자료 RAG의 **질문 → 검색 → 답변 → 출처** 흐름을 안정화하고, **무관 질문 차단·다턴·품질 실험·하이브리드 검색**까지 백엔드에서 묶었습니다.  
프론트(UI)는 건드리지 않았고, **API와 `.env.example`**, Streamlit **최소 연동**(다턴·스트리밍 데모)만 있습니다.

**이번 PR/커밋에 권장 메시지:**

```text
feat: RAG 백엔드 고도화 — 검색·다턴·품질실험·하이브리드·parent-child

- 문서 로더/인덱스 API, 관련도 거부·dedup·retrieval 메타
- 다턴 대화·후속 질문 검색 보강, 품질 실험 스크립트
- BM25+벡터 RRF, rerank, context 압축, parent-child 청킹
- Chroma 싱글톤, LLM 스트리밍 API, .env.example
```

---

## 왜 이렇게까지 했는지

프로젝트 성격이 **「RAG를 다루는 서비스」**라서, UI보다 먼저 아래가 맞아야 한다고 봤습니다.

1. **검색이 틀리면** 답도 틀리고, 토큰만 낭비됩니다.  
2. **환각·무관 답변**은 강의 도우미에서 치명적입니다.  
3. 팀이 나눠 일하려면 **프론트가 붙을 API**와 **품질을 말할 근거**(`retrieval`, 실험 스크립트)가 필요합니다.

README 「향후」는 팀 정책상 **수정하지 않았습니다.** 이 보고서와 코드·`.env.example`이 실제 구현 기준입니다.

---

## 이미 원격에 있던 것 (참고)

`dev-hm`에 이전에 올라간 커밋들도 같은 줄기입니다.

| 커밋 요지 | 의미 |
|-----------|------|
| `fix: RAG 검색 k 연동 및 관련 청크 필터, 프롬프트 강화` | k·threshold·프롬프트로 검색·답 품질 기반 |
| `fix: 개념 설명 모드 프롬프트 환각·제목 오해 방지` | 슬라이드/파일명을 정의로 쓰는 환각 완화 |

**이번에 로컬에 모아 커밋할 분량**은 그 위에 **로더·벡터 API·다턴·고급 검색·품질 실험**을 한 번에 올리는 단위입니다.

---

## 이번에 넣는 작업 — 의미 / 목적 / 기대 효과

### 1) 문서 로딩·메타 (`loaders.py`)

- **의미:** PDF/TXT뿐 아니라 **DOCX·PPTX** 지원, 모든 청크에 `subject`, `file_name`, `file_path` 등 메타 일관 적용.  
- **목적:** 예전 버그(마지막 파일만 메타 적용) 제거, 과목·파일 필터·출처 표시의 기반.  
- **기대 효과:** 재인덱싱 후 검색·출처가 파일/과목 단위로 맞음.  
- **팀 주의:** 로더·형식 변경 후 **재인덱싱** 필요.

### 2) 검색·답변 코어 (`retriever.py`, `rag_chain.py`, `generator.py`)

- **의미:** 유사도 검색 → **관련도 임계값** → 파일당 1청크 **dedup** → LLM. 무관하면 **LLM 호출 없이** 거부.  
- **목적:** 비용·속도·환각을 동시에 줄이는 1차 방어선.  
- **기대 효과:** 「날씨」「라면」 같은 질문은 빠르게 거부, 강의 질문은 근거 있는 답·출처.  
- **추가:** `result["retrieval"]`로 `best_score`, `context_chunk_count`, `rejection_reason` 등 **품질 설명 가능**.

### 3) 벡터 DB·운영 API (`vector_store.py`)

- **의미:** `index_data_dir`, `ingest_uploaded_files`, `list_indexed_documents`, `delete_indexed_document`, `get_index_stats`. Windows **WinError 32** 시 컬렉션 삭제 우선.  
- **목적:** 프론트·관리 기능이 붙을 **백엔드 엔드포인트** 확보.  
- **기대 효과:** 업로드·문서 삭제 UI는 프론트가 API만 호출하면 됨.

### 4) 다턴 대화 + 후속 질문 검색 (`rag_chain`, `app.py` 최소)

- **의미:** 매 턴 **다시 Chroma 검색**. `chat_history`는 맥락·짧은 후속 질문용 **query 보강**(이전 user + 현재 질문).  
- **목적:** 「방금 말한 Retrieval만 더」 같은 질문에서 검색 실패 줄이기.  
- **기대 효과:** 세션 안에서 이어 말하기 가능. **영구 저장**은 미구현(프론트·DB 영역).

### 5) 품질 실험·튜닝 (추천도 3) (`quality_experiments.py`)

- **의미:** §9 유형 질문 일괄 실행, **`--probe-only`**(LLM 없음), **threshold/k 스윕**, JSON 저장. `run_retrieval()` 분리.  
- **목적:** 「잘 된다」를 숫자·pass/fail로 말하기.  
- **기대 효과:** `.env`의 `RELEVANCE_SCORE_THRESHOLD`, `RETRIEVER_TOP_K`를 데이터에 맞게 조정 가능.  
- **담당자 실험:** `python -m src.quality_experiments --probe-only` → 스윕 → (선택) full 실험.

### 6) 고급 RAG (추천도 2) — 새 모듈

| 기능 | 파일 | 목적 | 기대 효과 |
|------|------|------|-----------|
| Chroma·임베딩 **싱글톤** | `vector_store.py` | 매 요청 객체 재생성 제거 | 지연·안정성 |
| **BM25 + 벡터 RRF** | `hybrid_search.py` | 키워드·의미 둘 다 | 용어·약어 질문 recall ↑ |
| **겹침 rerank** | `rerank.py` | fetch 넓힌 뒤 재정렬 | 관련 청크가 위로 |
| **extractive 압축** | `context_compress.py` | LLM에 넣는 길이 제한 | 토큰·비용 ↓ |
| **parent–child** | `splitter.py` | 작은 청크로 찾고, 페이지/슬라이드 **parent**로 읽기 | PPTX·긴 PDF 맥락 ↑ |
| **스트리밍 API** | `rag_chain.ask_question_events` | 토큰 단위 이벤트 | 프론트·체감 속도 (UI는 프론트) |

- **env:** `.env.example`에 위 기능 on/off·상수 정리. `config.py` 기본값과 동기.  
- **의존성:** `rank-bm25` (`requirements.txt`).

**중요:** parent–child·BM25는 **재인덱싱 후**에야 인덱스와 맞습니다. pull 받은 뒤 **한 번 인덱싱**해 주세요.

---

## 프론트 팀 — 하지 않은 것 / 붙일 것

| 항목 | 백엔드 | 프론트 |
|------|--------|--------|
| 파일 업로드 UI | `ingest_uploaded_files` | UI·저장 경로 |
| 문서 목록·삭제 | `list_*`, `delete_*` | 관리 화면 |
| `retrieval` 패널 | JSON 제공 | 표시·디버그 UI |
| 스트리밍 UI | `ask_question_events` | 채팅 렌더 |
| README 향후 문구 | 미수정 | README 갱신은 팀 합의 시 |

`app.py` 변경은 **다턴·스트리밍 데모** 수준이며, **제품 UI 작업은 프론트**입니다.

---

## pull 받은 뒤 할 일 (체크리스트)

1. `pip install -r requirements.txt` (`rank-bm25` 포함)  
2. `copy .env.example .env` 후 키 입력 (기존 `.env` 유지 시 **새 변수**는 example 참고)  
3. Streamlit **재시작**  
4. 사이드바 **문서 인덱싱** (가능하면 「벡터 DB 초기화」 후 — parent–child·BM25 반영)  
5. `python -m src.quality_experiments --probe-only`  
6. (선택) `--sweep-threshold` → 값 확정 후 full 실험  

**WinError 32** 나오면: Streamlit 끄고 다시 인덱싱 (기존과 동일).

---

## 점검해 둔 것 / 남은 리스크

**해 둔 것**

- `src` 문법 컴파일  
- probe 품질 실험: 자동 케이스 **pass 4 / fail 0** (threshold 1.2, hybrid/rerank on)  
- 싱글톤·스트리밍 스모크  

**남은 것 (버그라기보다 운영·범위)**

- **재인덱싱 전** parent–child 효과 제한적  
- full LLM 품질 실험·§10 수동 평가 → 담당자 실행  
- 대용량 코퍼스 시 BM25 메모리 — 현재 강의 규모 가정  
- 추천도 1 (MMR, Self-RAG 등) — **미구현**

---

## 파일 변경 요약 (리뷰용)

| 영역 | 주요 파일 |
|------|-----------|
| 신규 | `.env.example`, `quality_experiments.py`, `hybrid_search.py`, `rerank.py`, `context_compress.py` |
| 수정 | `loaders.py`, `vector_store.py`, `rag_chain.py`, `retriever.py`, `splitter.py`, `config.py`, `generator.py`, `app.py`, `requirements.txt` |
| Git | `docs/TEAM_DELIVERY_REPORT.md` 만 추적 (`.gitignore` 예외) |

---

## 문의 시 참고 API

```python
from src.rag_chain import ask_question, ask_question_events, run_retrieval
from src.vector_store import index_data_dir, ingest_uploaded_files, list_indexed_documents, delete_indexed_document
```

- 일반 답변: `ask_question(질문, 모드, chat_history=..., subject=..., file_name=...)`  
- 스트리밍: `for e in ask_question_events(...):` → `token` / `final`  
- 검색만: `run_retrieval(질문)` (튜닝·probe)

---

README와 다른 부분은 **이 보고서·코드·`.env.example`** 을 우선해 주시면 됩니다.
