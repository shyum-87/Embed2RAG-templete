#!/usr/bin/env python3
"""Streamlit chatbot using external RAG API + LLM API."""
from __future__ import annotations

import streamlit as st

from api_clients import (
    LlmApiConfig,
    RagApiConfig,
    call_llm_chat_api,
    list_rags_from_api,
    retrieve_contexts,
)


def main() -> None:
    st.set_page_config(page_title="RAG2Chatbot API", layout="wide")
    st.title("🌐 RAG2Chatbot (API Version)")
    st.caption("외부 RAG API + 외부 LLM API 조합 버전")

    with st.sidebar:
        st.header("RAG API 설정")
        rag_url = st.text_input("RAG API Base URL", value="http://localhost:8000")
        rag_key = st.text_input("RAG API Key", value="", type="password")
        rag_key_header = st.text_input("RAG Key Header", value="Authorization")
        rag_key_prefix = st.text_input("RAG Key Prefix", value="Bearer")

        st.header("LLM API 설정")
        llm_url = st.text_input("LLM API Base URL", value="http://localhost:11434")
        llm_key = st.text_input("LLM API Key", value="", type="password")
        llm_model = st.text_input("LLM Model", value="gpt-4o-mini")
        llm_key_header = st.text_input("LLM Key Header", value="Authorization")
        llm_key_prefix = st.text_input("LLM Key Prefix", value="Bearer")

        top_k = st.slider("Top K", min_value=1, max_value=20, value=4)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)
        max_tokens = st.slider("Max Tokens", min_value=64, max_value=2048, value=512, step=64)

        system_prompt = st.text_area(
            "System Prompt",
            value="당신은 RAG 기반 어시스턴트다. 주어진 컨텍스트에 근거해 정확히 답변하라.",
            height=100,
        )

        load_rags_btn = st.button("RAG 목록 불러오기")

    if "rag_names" not in st.session_state:
        st.session_state.rag_names = []

    rag_cfg = RagApiConfig(
        base_url=rag_url,
        api_key=rag_key,
        api_key_header=rag_key_header,
        api_key_prefix=rag_key_prefix,
    )

    llm_cfg = LlmApiConfig(
        base_url=llm_url,
        api_key=llm_key,
        model=llm_model,
        api_key_header=llm_key_header,
        api_key_prefix=llm_key_prefix,
    )

    if load_rags_btn:
        try:
            st.session_state.rag_names = list_rags_from_api(rag_cfg)
            st.success(f"RAG {len(st.session_state.rag_names)}개 로드")
        except Exception as e:
            st.error(f"RAG 목록 조회 실패: {e}")

    manual_rags = st.text_input("RAG 이름 수동 입력(쉼표구분)", value="")
    selected_rags = st.multiselect("RAG 선택(중복 선택 허용)", st.session_state.rag_names)
    if manual_rags.strip():
        selected_rags.extend([x.strip() for x in manual_rags.split(",") if x.strip()])

    # dedup while keeping order
    selected_rags = list(dict.fromkeys(selected_rags))

    st.subheader("채팅")
    question = st.text_area("질문", placeholder="예) 2024년 매출 요약해줘", height=120)
    ask_btn = st.button("질문하기", type="primary")

    if ask_btn:
        if not question.strip():
            st.warning("질문을 입력하세요.")
            return
        if not selected_rags:
            st.warning("최소 1개 RAG를 선택하거나 수동 입력하세요.")
            return

        try:
            with st.spinner("RAG API 검색 중..."):
                contexts = retrieve_contexts(rag_cfg, question, selected_rags, top_k=top_k)

            with st.spinner("LLM API 답변 생성 중..."):
                resp = call_llm_chat_api(
                    cfg=llm_cfg,
                    system_prompt=system_prompt,
                    user_question=question,
                    contexts=contexts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            st.markdown("### 답변")
            st.write(resp["answer"])

            with st.expander("RAG 컨텍스트"):
                if not contexts:
                    st.info("검색된 컨텍스트가 없습니다.")
                for i, c in enumerate(contexts, 1):
                    st.markdown(f"**[{i}] source:** {c.get('source', '')}")
                    st.write(c.get("text", ""))
                    if c.get("metadata"):
                        st.caption(str(c["metadata"]))

        except Exception as e:
            st.error(f"질문 처리 실패: {e}")


if __name__ == "__main__":
    main()
