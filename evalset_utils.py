#!/usr/bin/env python3
"""Utilities for building evaluation datasets from existing RAG artifacts."""
from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class EvalSample:
    question: str
    gold_answer: str
    gold_sources: List[str]
    rag_name: str
    record_id: str


def list_rags(base_dir: str) -> List[str]:
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    return sorted([d.name for d in p.iterdir() if d.is_dir()])


def list_chunk_files(base_dir: str, rag_name: str) -> List[pathlib.Path]:
    p = pathlib.Path(base_dir) / rag_name / "chunks"
    if not p.exists():
        return []
    return sorted(p.glob("*.json"))


def list_raw_files(base_dir: str, rag_name: str) -> List[pathlib.Path]:
    p = pathlib.Path(base_dir) / rag_name / "raw_json"
    if not p.exists():
        return []
    return sorted(p.glob("*.json"))


def _load_json_array(path: pathlib.Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"JSON array expected: {path}")
    return obj


def iter_chunk_records(base_dir: str, rag_names: List[str]) -> Iterable[Dict[str, Any]]:
    for rag in rag_names:
        for f in list_chunk_files(base_dir, rag):
            for row in _load_json_array(f):
                row["_rag_name"] = rag
                row["_source_file"] = str(f)
                yield row


def iter_raw_records(base_dir: str, rag_names: List[str]) -> Iterable[Dict[str, Any]]:
    for rag in rag_names:
        for f in list_raw_files(base_dir, rag):
            for row in _load_json_array(f):
                row["_rag_name"] = rag
                row["_source_file"] = str(f)
                yield row


def _first_sentence(text: str, max_len: int = 120) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t:
        return ""
    parts = t.split(".")
    sent = parts[0].strip() if parts else t
    if len(sent) > max_len:
        sent = sent[:max_len].rstrip() + "..."
    return sent


def _question_from_title(title: str, preview: str) -> str:
    if title:
        return f"{title} 문서에서 핵심 내용을 설명해줘."
    if preview:
        return f"다음 내용과 관련된 핵심은 무엇인가? ({preview})"
    return "이 문서의 핵심 내용을 설명해줘."


def build_eval_samples_from_chunks(
    base_dir: str,
    rag_names: List[str],
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[EvalSample]:
    rows = list(iter_chunk_records(base_dir, rag_names))
    if not rows:
        return []

    if sample_size is not None and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)

    out: List[EvalSample] = []
    for row in rows:
        md = row.get("metadata", {}) or {}
        text = row.get("text", "")
        title = str(md.get("title", ""))
        source_path = str(md.get("source_path", ""))
        preview = _first_sentence(text)

        out.append(
            EvalSample(
                question=_question_from_title(title, preview),
                gold_answer=text,
                gold_sources=[source_path] if source_path else [],
                rag_name=row.get("rag_name", row.get("_rag_name", "")),
                record_id=row.get("chunk_id", ""),
            )
        )
    return out


def build_eval_samples_from_raw_json(
    base_dir: str,
    rag_names: List[str],
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[EvalSample]:
    rows = list(iter_raw_records(base_dir, rag_names))
    if not rows:
        return []

    if sample_size is not None and sample_size < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, sample_size)

    out: List[EvalSample] = []
    for i, row in enumerate(rows):
        text = row.get("text", "")
        title = str(row.get("title", ""))
        source_path = str(row.get("source_path", ""))
        preview = _first_sentence(text)

        out.append(
            EvalSample(
                question=_question_from_title(title, preview),
                gold_answer=text,
                gold_sources=[source_path] if source_path else [],
                rag_name=row.get("_rag_name", ""),
                record_id=f"raw_{i}",
            )
        )
    return out


def save_jsonl(samples: List[EvalSample], out_path: str) -> None:
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            obj = {
                "question": s.question,
                "gold_answer": s.gold_answer,
                "gold_sources": s.gold_sources,
                "rag_name": s.rag_name,
                "record_id": s.record_id,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
