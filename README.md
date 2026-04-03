# Embed2RAG-templete

LangChain 기반으로 **임베딩 파이프라인(Embeded2RAG.py)** 과 **다중 RAG 챗봇 UI(RAG2Chatbot.py)** 를 분리한 템플릿입니다.

## 1) Embeded2RAG.py

기능:
- Excel / PPT / SQLite Table 입력을 JSON 정규화
- Chunking 수행
- `rag_name` 기준으로 주제별 저장
- 임베딩 모델/벡터 DB 교체 가능 구조

예시:
```bash
python Embeded2RAG.py \
  --rag-name finance \
  --excel ./data/sales.xlsx \
  --ppt ./data/briefing.pptx \
  --sqlite-db ./data/app.db \
  --db-table orders customers \
  --embed-kind hf-local \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-store chroma
```

RAG 목록 확인:
```bash
python Embeded2RAG.py --rag-name dummy --list-rags
```

> 참고: 아직 벡터 DB/모델이 미정이면 `--dry-run`으로 JSON+Chunk 생성만 먼저 수행할 수 있습니다.

## 2) RAG2Chatbot.py

기능:
- Streamlit 채팅 UI
- RAG 목록 표시 + 체크박스 다중 선택
- 선택된 여러 RAG를 하나의 Retriever로 병합

실행:
```bash
streamlit run RAG2Chatbot.py
```

사이드바에서:
- RAG 저장 경로
- Embedding 모델(HF ID 또는 로컬 경로)
- LLM 종류/모델
- top_k
- 체크박스로 다중 RAG 선택

## 모델/DB 교체 포인트

- `Embeded2RAG.py`
  - `build_embeddings()`
  - `build_vector_store()`
- `RAG2Chatbot.py`
  - `build_embeddings()`
  - `build_llm()`
  - `get_vectorstore_for_rag()`

중국 내에서 다운로드 가능한 모델/DB 확정 후 위 함수만 교체하면 전체 로직 재사용이 가능합니다.
