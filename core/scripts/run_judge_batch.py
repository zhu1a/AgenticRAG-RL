#!/usr/bin/env python3
"""Run the three LLM-judge metrics on one or more evaluation result files.

This is the batch companion expected by ``training/sft_pipeline.sh``.  It
accepts the result shape produced by ``evaluation/run_eval.py`` and also the
slightly older ``run_cloud_eval.py`` field names.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_gold(data_dir: str):
    with open(os.path.join(data_dir, "qa_pairs.json"), encoding="utf-8") as f:
        qa_items = json.load(f)
    with open(os.path.join(data_dir, "corpus.json"), encoding="utf-8") as f:
        corpus = {item["chunk_id"]: item for item in json.load(f)}
    return {item["final_question"]: item for item in qa_items}, corpus


def _gold_documents(qa: dict, corpus: dict) -> str:
    parts = []
    for hop in qa.get("hops", []):
        cid = hop.get("doc_chunk_id", "")
        doc = corpus.get(cid, {})
        text = doc.get("text", hop.get("answer", ""))
        title = doc.get("title", "")
        parts.append(f"[{cid}] {title}\n{text}")
    return "\n\n".join(parts)


def _judge_one(record: dict, qa_map: dict, corpus: dict, lang: str):
    from evaluation.llm_judge import (
        judge_answer_correctness,
        judge_context_precision,
        judge_faithfulness,
    )

    question = record.get("question", "")
    prediction = record.get("prediction", record.get("pred", ""))
    gold = record.get("gold", "")
    evidence = record.get("evidence_text", "")
    qa = qa_map.get(question, {})

    judged = dict(record)
    judged["judge_correctness"] = round(
        judge_answer_correctness(question, prediction, gold, lang=lang), 3
    )
    if evidence.strip():
        judged["judge_faithfulness"] = round(
            judge_faithfulness(question, prediction, evidence, lang=lang), 3
        )
        gold_docs = _gold_documents(qa, corpus)
        if gold_docs:
            judged["judge_ctx_precision"] = round(
                judge_context_precision(question, evidence, gold_docs, lang=lang), 3
            )
    return judged


def judge_file(path: str, qa_map: dict, corpus: dict, workers: int, lang: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    records_key = "results" if "results" in data else "per_sample"
    records = data.get(records_key, [])
    judged = [None] * len(records)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_judge_one, record, qa_map, corpus, lang): index
            for index, record in enumerate(records)
        }
        for done, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                judged[index] = future.result()
            except Exception as exc:
                judged[index] = dict(records[index])
                judged[index]["judge_error"] = str(exc)
            if done == 1 or done % 20 == 0 or done == len(records):
                print(f"  [{done}/{len(records)}] judged")

    data[records_key] = judged
    summary = data.setdefault("summary", {})
    for field, summary_field in [
        ("judge_correctness", "avg_judge_correctness"),
        ("judge_faithfulness", "avg_judge_faithfulness"),
        ("judge_ctx_precision", "avg_judge_ctx_precision"),
    ]:
        scores = [r[field] for r in judged if field in r]
        if scores:
            summary[summary_field] = round(sum(scores) / len(scores), 3)

    stem, ext = os.path.splitext(path)
    output = f"{stem}_judged{ext}"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  saved: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Batch LLM Judge for evaluation files")
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lang", choices=["en", "zh"], default="zh")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    qa_map, corpus = _load_gold(args.data_dir)
    print(f"[judge] loaded {len(qa_map)} questions and {len(corpus)} chunks")
    for path in args.result_files:
        print(f"[judge] {path}")
        judge_file(path, qa_map, corpus, args.workers, args.lang)


if __name__ == "__main__":
    main()
