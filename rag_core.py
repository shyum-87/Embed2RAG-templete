#!/usr/bin/env python3
"""Shared core utilities for multi-RAG retrieval and QA chain building."""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import List

from langchain.chains import RetrievalQA
from langchain.schema import BaseRetriever
from langchain_community.vectorstores import Chroma


@dataclass
class AppConfig:
    base_dir: str = "./rag_store"
    vector_store_name: str = "chroma"
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
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name_or_path)


def build_llm(kind: str, model_name_or_path: str):
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

        seen = set()
        dedup = []
        for d in docs:
            key = (d.page_content, str(d.metadata.get("source_path", "")))
            if key not in seen:
                seen.add(key)
                dedup.append(d)
        return dedup


def build_multi_retriever(config: AppConfig, selected_rags: List[str], embed_model: str) -> MultiRetriever:
    embeddings = build_embeddings(embed_model)
    retrievers = []
    for rag_name in selected_rags:
        vs = get_vectorstore_for_rag(config.base_dir, rag_name, embeddings)
        retrievers.append(vs.as_retriever(search_kwargs={"k": config.k}))
    return MultiRetriever(retrievers)


def build_multi_rag_chain(
    config: AppConfig,
    selected_rags: List[str],
    embed_model: str,
    llm_kind: str,
    llm_model: str,
):
    merged = build_multi_retriever(config=config, selected_rags=selected_rags, embed_model=embed_model)
    llm = build_llm(llm_kind, llm_model)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=merged,
        return_source_documents=True,
    )
