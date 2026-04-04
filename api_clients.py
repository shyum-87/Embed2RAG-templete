#!/usr/bin/env python3
"""HTTP clients for external RAG API and LLM API backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class RagApiConfig:
    base_url: str
    api_key: str = ""
    timeout_sec: int = 60
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"


@dataclass
class LlmApiConfig:
    base_url: str
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_sec: int = 120
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"


def _build_headers(api_key: str, key_header: str, key_prefix: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        if key_prefix:
            headers[key_header] = f"{key_prefix} {api_key}".strip()
        else:
            headers[key_header] = api_key
    return headers


def list_rags_from_api(cfg: RagApiConfig) -> List[str]:
    url = f"{cfg.base_url.rstrip('/')}/rags"
    resp = requests.get(
        url,
        headers=_build_headers(cfg.api_key, cfg.api_key_header, cfg.api_key_prefix),
        timeout=cfg.timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        items = data.get("rags") or data.get("items") or []
        if isinstance(items, list):
            return [str(x.get("name", x)) if isinstance(x, dict) else str(x) for x in items]

    return []


def retrieve_contexts(
    cfg: RagApiConfig,
    query: str,
    rag_names: List[str],
    top_k: int = 4,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Expected request/response (권장 형식)
    Request POST /retrieve
    {
      "query": "...",
      "rag_names": ["finance", "hr"],
      "top_k": 4,
      "filters": {...}
    }

    Response JSON
    {
      "contexts": [
        {"text": "...", "source": "...", "metadata": {...}},
        ...
      ]
    }
    """
    url = f"{cfg.base_url.rstrip('/')}/retrieve"
    payload = {
        "query": query,
        "rag_names": rag_names,
        "top_k": top_k,
        "filters": extra_filters or {},
    }

    resp = requests.post(
        url,
        json=payload,
        headers=_build_headers(cfg.api_key, cfg.api_key_header, cfg.api_key_prefix),
        timeout=cfg.timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list):
        # allow raw list response
        return data

    if isinstance(data, dict):
        contexts = data.get("contexts") or data.get("documents") or data.get("results") or []
        if isinstance(contexts, list):
            return contexts

    return []


def call_llm_chat_api(
    cfg: LlmApiConfig,
    system_prompt: str,
    user_question: str,
    contexts: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Dict[str, Any]:
    """
    OpenAI-compatible /v1/chat/completions 형식 사용.
    다른 API라면 해당 함수만 변경하면 됩니다.
    """
    url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"

    context_block = "\n\n".join(
        [f"[CTX {i+1}] {c.get('text', '')}" for i, c in enumerate(contexts)]
    )

    prompt = (
        f"질문: {user_question}\n\n"
        f"아래 컨텍스트만 근거로 답변하세요. 근거가 없으면 모른다고 답하세요.\n"
        f"컨텍스트:\n{context_block}"
    )

    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(
        url,
        json=payload,
        headers=_build_headers(cfg.api_key, cfg.api_key_header, cfg.api_key_prefix),
        timeout=cfg.timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()

    answer = ""
    try:
        answer = data["choices"][0]["message"]["content"]
    except Exception:
        answer = str(data)

    return {"answer": answer, "raw": data}
