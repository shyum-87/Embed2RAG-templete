#!/usr/bin/env python3
"""
RAG2Chatbot.py

요구사항 반영:
- 채팅 UI 제공
- RAG 목록을 체크박스(중복 선택)로 선택
- 선택된 다중 RAG를 동시에 Retriever로 묶어서 답변
- LLM/임베딩/벡터DB는 나중에 교체 가능하게 구성

실행 예:
streamlit run RAG2Chatbot.py
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import List

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_community.vectorstores import Chroma


@dataclass
class AppConfig:
    base_dir: str = "./rag_store"
    vector_store_name: str = "chroma"  # reserved for future
    k: int = 4


def list_rags(base_dir: str) -> List[str]:
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def get_vectorstore_for_rag(base_dir: str, rag_name: str, embeddings):
    vectordb_dir = os.path.join(base_dir, rag_name, "vectordb")
    if not os.path.exists(vectordb_dir):
        raise FileNotFoundError(f"vector db 없음: {vectordb_dir}")

    return Chroma(
        embedding_function=embeddings,
        persist_directory=vectordb_dir,
    )


def build_embeddings(model_name_or_path: str):
    # 추후 중국 내 배포 모델로 쉽게 교체 가능
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name_or_path)


def build_llm(kind: str, model_name_or_path: str):
    """
    LLM 교체 포인트.

    기본은 로컬 transformers pipeline을 사용.
    운영 시 vLLM, Ollama, OpenAI-compatible endpoint 등으로 변경 가능.
    """
    kind = kind.lower().strip()

    if kind == "hf-pipeline":
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        from langchain_community.llms import HuggingFacePipeline

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError("현재 지원 LLM kind: hf-pipeline")


class MultiRetriever(BaseRetriever):
    def __init__(self, retrievers: List[BaseRetriever]):
        super().__init__()
        self.retrievers = retrievers

    def _get_relevant_documents(self, query: str):
        docs = []
        for r in self.retrievers:
            docs.extend(r.get_relevant_documents(query))

        # 중복 제거 (content + source 기준)
        seen = set()
        dedup = []
        for d in docs:
            key = (d.page_content, str(d.metadata.get("source_path", "")))
            if key not in seen:
                seen.add(key)
                dedup.append(d)
        return dedup


def build_multi_rag_chain(config: AppConfig, selected_rags: List[str], embed_model: str, llm_kind: str, llm_model: str):
    embeddings = build_embeddings(embed_model)
    retrievers = []
    for rag_name in selected_rags:
        vs = get_vectorstore_for_rag(config.base_dir, rag_name, embeddings)
        retrievers.append(vs.as_retriever(search_kwargs={"k": config.k}))

    merged = MultiRetriever(retrievers)
    llm = build_llm(llm_kind, llm_model)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=merged,
        return_source_documents=True,
    )


def main() -> None:
    st.set_page_config(page_title="RAG2Chatbot", layout="wide")
    st.title("🧠 RAG2Chatbot")
    st.caption("다중 RAG 선택(체크박스) + LangChain 기반 챗봇")

    config = AppConfig()

    with st.sidebar:
        st.header("설정")
        base_dir = st.text_input("RAG 저장 경로", value=config.base_dir)
        embed_model = st.text_input("Embedding 모델(HF ID/Local)", value="sentence-transformers/all-MiniLM-L6-v2")
        llm_kind = st.selectbox("LLM 종류", options=["hf-pipeline"], index=0)
        llm_model = st.text_input("LLM 모델(HF ID/Local)", value="gpt2")
        top_k = st.slider("Retriever top_k", min_value=1, max_value=10, value=4)

        config.base_dir = base_dir
        config.k = top_k

        rag_names = list_rags(base_dir)
        st.subheader("RAG 목록 (중복 선택 허용)")

        selected_rags = []
        for rag in rag_names:
            checked = st.checkbox(rag, value=False)
            if checked:
                selected_rags.append(rag)

        run_btn = st.button("체인 로드", type="primary")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    if run_btn:
        if not selected_rags:
            st.warning("최소 1개 RAG를 선택하세요.")
        else:
            with st.spinner("RAG 체인 로드 중..."):
                try:
                    st.session_state.qa_chain = build_multi_rag_chain(
                        config=config,
                        selected_rags=selected_rags,
                        embed_model=embed_model,
                        llm_kind=llm_kind,
                        llm_model=llm_model,
                    )
                    st.success(f"로드 완료: {', '.join(selected_rags)}")
                except Exception as e:
                    st.error(f"체인 로드 실패: {e}")

    st.subheader("채팅")
    query = st.text_area("질문을 입력하세요", height=120, placeholder="예) 2024년 매출 동향 알려줘")
    ask_btn = st.button("질문하기")

    if ask_btn:
        if not st.session_state.qa_chain:
            st.warning("먼저 사이드바에서 RAG를 선택하고 '체인 로드'를 눌러주세요.")
        elif not query.strip():
            st.warning("질문을 입력하세요.")
        else:
            with st.spinner("답변 생성 중..."):
                try:
                    result = st.session_state.qa_chain({"query": query})
                    answer = result.get("result", "")
                    sources = result.get("source_documents", [])

                    st.markdown("### 답변")
                    st.write(answer)

                    with st.expander("근거 문서"):
                        for i, d in enumerate(sources, 1):
                            st.markdown(f"**[{i}]** {d.metadata}")
                            st.write(d.page_content[:700] + ("..." if len(d.page_content) > 700 else ""))
                except Exception as e:
                    st.error(f"질문 처리 실패: {e}")


if __name__ == "__main__":
    main()
