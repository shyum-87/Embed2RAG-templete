# Embed2RAG-templete

LangChain 기반으로 **임베딩 파이프라인(Embeded2RAG.py)** 과 **다중 RAG 챗봇 UI(RAG2Chatbot.py)** 를 분리한 템플릿입니다.

## 1) Embeded2RAG.py

기능:
- Excel / PPT / PDF / Word / SQLite Table 입력을 JSON 정규화
- Chunking 수행
- `rag_name` 기준으로 주제별 저장
- 임베딩 모델/벡터 DB 교체 가능 구조

예시:
```bash
python Embeded2RAG.py \
  --rag-name finance \
  --excel ./data/sales.xlsx \
  --ppt ./data/briefing.pptx \
  --pdf ./data/policy.pdf \
  --word ./data/faq.docx \
  --sqlite-db ./data/app.db \
  --db-table orders customers \
  --embed-kind hf-local \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-store chroma
```

Qdrant 예시:
```bash
python Embeded2RAG.py \
  --rag-name finance \
  --excel ./data/sales.xlsx \
  --embed-kind hf-local \
  --embed-model BAAI/bge-m3 \
  --vector-store qdrant \
  --qdrant-url http://127.0.0.1:6333 \
  --qdrant-collection finance
```

## 2) RAG2Chatbot.py (로컬 모델/로컬 벡터DB 버전)

실행:
```bash
streamlit run RAG2Chatbot.py
```

## 3) RAG2Chatbot_API.py (RAG API + LLM API 버전)

요구사항 반영:
- 로컬 벡터DB 대신 외부 **RAG API** 호출 (`/rags`, `/retrieve`)
- 로컬 LLM 대신 외부 **LLM API(OpenAI-compatible)** 호출 (`/v1/chat/completions`)
- RAG 다중 선택(또는 수동 입력) 지원

실행:
```bash
streamlit run RAG2Chatbot_API.py
```

권장 RAG API 응답 예시:
```json
{
  "contexts": [
    {"text": "...", "source": "finance.xlsx", "metadata": {"sheet": "Q1"}}
  ]
}
```

## 4) EvalsetBuilder.py (평가셋 생성 앱)

요구사항 반영:
- **RAG에 저장된 chunks를 순서/샘플 기반으로 불러와 평가셋 생성**
- **RAG 입력 전 단계(raw_json) 파일 기반으로 평가셋 생성**
- 생성 후 UI에서 질문/정답/소스 편집 가능
- JSONL로 저장 후 `eval_rag.py`에 바로 투입 가능

실행:
```bash
streamlit run EvalsetBuilder.py
```

## 5) eval_rag.py (평가 실행)

평가셋(JSONL 또는 JSON 배열) 실행 예:
```bash
python eval_rag.py \
  --dataset ./evaluation/generated_evalset.jsonl \
  --base-dir ./rag_store \
  --rags finance hr \
  --embed-model sentence-transformers/all-MiniLM-L6-v2 \
  --llm-kind hf-pipeline \
  --llm-model gpt2 \
  --k 4 \
  --out ./evaluation/report.csv
```

## 모델/DB/API 교체 포인트

- `Embeded2RAG.py`
  - `build_embeddings()`
  - `build_vector_store()`
- `rag_core.py`
  - `build_embeddings()`
  - `build_llm()`
  - `get_vectorstore_for_rag()`
- `api_clients.py`
  - `retrieve_contexts()` (RAG API 계약)
  - `call_llm_chat_api()` (LLM API 계약)

중국 내에서 사용 가능한 인프라 확정 후 위 함수만 교체하면 전체 로직 재사용이 가능합니다.
