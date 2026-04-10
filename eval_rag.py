#!/usr/bin/env python3
"""
Evaluate multi-RAG chatbot quality.

Input file format (JSONL recommended):
{"question":"...","gold_answer":"...","gold_sources":["/path/a.xlsx","/path/b.pptx"]}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from rag_core import AppConfig, build_multi_rag_chain, build_multi_retriever


@dataclass
class EvalItem:
    question: str
    gold_answer: str
    gold_sources: List[str]


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def token_set(s: str) -> set[str]:
    return set(re.findall(r"\w+", normalize_text(s)))


def answer_similarity(pred: str, gold: str) -> float:
    if not pred and not gold:
        return 1.0
    return SequenceMatcher(None, normalize_text(pred), normalize_text(gold)).ratio()


def token_f1(pred: str, gold: str) -> float:
    p = token_set(pred)
    g = token_set(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = len(p & g)
    precision = overlap / len(p)
    recall = overlap / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def source_recall_at_k(retrieved_sources: List[str], gold_sources: List[str], k: int) -> float:
    gold = {normalize_text(x) for x in gold_sources if x}
    if not gold:
        return 1.0
    topk = {normalize_text(x) for x in retrieved_sources[:k] if x}
    return len(gold & topk) / len(gold)


def faithfulness_overlap(answer: str, context_docs: List[str]) -> float:
    """
    Heuristic groundedness score:
    answer token 중 context token에 포함되는 비율.
    """
    a = token_set(answer)
    if not a:
        return 0.0
    c = set()
    for d in context_docs:
        c |= token_set(d)
    if not c:
        return 0.0
    return len(a & c) / len(a)


def iter_eval_items(path: Path) -> Iterable[EvalItem]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield EvalItem(
                    question=obj["question"],
                    gold_answer=obj.get("gold_answer", ""),
                    gold_sources=obj.get("gold_sources", []),
                )
        return

    # fallback: JSON array
    with path.open("r", encoding="utf-8") as f:
        arr = json.load(f)
    for obj in arr:
        yield EvalItem(
            question=obj["question"],
            gold_answer=obj.get("gold_answer", ""),
            gold_sources=obj.get("gold_sources", []),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RAG quality")
    p.add_argument("--dataset", required=True, help="평가셋 json/jsonl 파일")
    p.add_argument("--base-dir", default="./rag_store")
    p.add_argument("--rags", nargs="+", required=True, help="평가할 RAG 이름들")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--llm-kind", default="hf-pipeline")
    p.add_argument("--llm-model", default="gpt2")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default="./eval_report.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    items = list(iter_eval_items(dataset_path))
    if not items:
        raise RuntimeError("평가셋이 비어 있습니다.")

    config = AppConfig(base_dir=args.base_dir, k=args.k)
    retriever = build_multi_retriever(config=config, selected_rags=args.rags, embed_model=args.embed_model)
    qa_chain = build_multi_rag_chain(
        config=config,
        selected_rags=args.rags,
        embed_model=args.embed_model,
        llm_kind=args.llm_kind,
        llm_model=args.llm_model,
    )

    rows: List[Dict[str, Any]] = []

    for i, item in enumerate(items, 1):
        t0 = time.perf_counter()

        docs = retriever.get_relevant_documents(item.question)
        retrieved_sources = [str(d.metadata.get("source_path", "")) for d in docs]
        source_recall = source_recall_at_k(retrieved_sources, item.gold_sources, args.k)

        result = qa_chain({"query": item.question})
        pred_answer = result.get("result", "")
        src_docs = result.get("source_documents", [])
        src_texts = [d.page_content for d in src_docs]

        sim = answer_similarity(pred_answer, item.gold_answer)
        f1 = token_f1(pred_answer, item.gold_answer)
        faith = faithfulness_overlap(pred_answer, src_texts)
        latency = time.perf_counter() - t0

        rows.append(
            {
                "idx": i,
                "question": item.question,
                "gold_answer": item.gold_answer,
                "pred_answer": pred_answer,
                "source_recall@k": round(source_recall, 4),
                "answer_similarity": round(sim, 4),
                "token_f1": round(f1, 4),
                "faithfulness_overlap": round(faith, 4),
                "latency_sec": round(latency, 4),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "count": len(rows),
        "avg_source_recall@k": round(mean(r["source_recall@k"] for r in rows), 4),
        "avg_answer_similarity": round(mean(r["answer_similarity"] for r in rows), 4),
        "avg_token_f1": round(mean(r["token_f1"] for r in rows), 4),
        "avg_faithfulness_overlap": round(mean(r["faithfulness_overlap"] for r in rows), 4),
        "avg_latency_sec": round(mean(r["latency_sec"] for r in rows), 4),
    }

    print("=== Eval Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"detail csv: {out_path}")


if __name__ == "__main__":
    main()
