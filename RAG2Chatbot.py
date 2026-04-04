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

import streamlit as st

from rag_core import AppConfig, build_multi_rag_chain, list_rags


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
