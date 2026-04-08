#!/usr/bin/env python3
"""Streamlit app for building evaluation datasets from RAG artifacts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from evalset_utils import (
    EvalSample,
    build_eval_samples_from_chunks,
    build_eval_samples_from_raw_json,
    list_rags,
    save_jsonl,
)


def samples_to_df(samples: list[EvalSample]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "question": s.question,
                "gold_answer": s.gold_answer,
                "gold_sources": ";".join(s.gold_sources),
                "rag_name": s.rag_name,
                "record_id": s.record_id,
            }
            for s in samples
        ]
    )


def df_to_samples(df: pd.DataFrame) -> list[EvalSample]:
    samples: list[EvalSample] = []
    for _, row in df.iterrows():
        sources = [x.strip() for x in str(row.get("gold_sources", "")).split(";") if x.strip()]
        samples.append(
            EvalSample(
                question=str(row.get("question", "")),
                gold_answer=str(row.get("gold_answer", "")),
                gold_sources=sources,
                rag_name=str(row.get("rag_name", "")),
                record_id=str(row.get("record_id", "")),
            )
        )
    return samples


def main() -> None:
    st.set_page_config(page_title="EvalsetBuilder", layout="wide")
    st.title("🧪 EvalsetBuilder")
    st.caption("RAG chunks/raw_json 기반으로 평가셋(JSONL)을 빠르게 생성")

    with st.sidebar:
        base_dir = st.text_input("RAG 저장 경로", value="./rag_store")
        rag_names = list_rags(base_dir)
        selected_rags = st.multiselect("대상 RAG", rag_names, default=rag_names[:1] if rag_names else [])
        source_mode = st.radio("생성 소스", options=["chunks", "raw_json"], horizontal=True)

        sample_size = st.number_input("샘플 개수 (0이면 전체)", min_value=0, value=200, step=10)
        seed = st.number_input("랜덤 시드", min_value=0, value=42, step=1)

        generate_btn = st.button("평가셋 생성", type="primary")

    if "eval_df" not in st.session_state:
        st.session_state.eval_df = pd.DataFrame(columns=["question", "gold_answer", "gold_sources", "rag_name", "record_id"])

    if generate_btn:
        if not selected_rags:
            st.warning("최소 1개 RAG를 선택하세요.")
        else:
            size = None if int(sample_size) == 0 else int(sample_size)
            if source_mode == "chunks":
                samples = build_eval_samples_from_chunks(base_dir, selected_rags, sample_size=size, seed=int(seed))
            else:
                samples = build_eval_samples_from_raw_json(base_dir, selected_rags, sample_size=size, seed=int(seed))

            st.session_state.eval_df = samples_to_df(samples)
            st.success(f"생성 완료: {len(samples)}개")

    st.subheader("평가셋 편집")
    edited = st.data_editor(st.session_state.eval_df, num_rows="dynamic", use_container_width=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        out_path = st.text_input("저장 경로(JSONL)", value="./evaluation/generated_evalset.jsonl")
    with c2:
        save_btn = st.button("JSONL 저장")

    if save_btn:
        samples = df_to_samples(edited)
        save_jsonl(samples, out_path)
        st.success(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
