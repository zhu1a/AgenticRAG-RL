#!/usr/bin/env python3
"""生成 Agentic GRPO 训练数据（Search-R1 风格）

将 oracle traces 转为 verl multi-turn agentic 格式：
- prompt: [system, user] 聊天消息（不含 evidence）
- 模型在 rollout 中自主生成 tool_call，环境执行检索返回结果
- reward 基于最终答案质量

用法:
  python scripts/prepare_agentic_grpo_data.py
"""
import json
import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = "你是一个金融文档问答 Agent。通过搜索相关文档来回答用户的问题。"

# 数据源路径（不依赖调用时的当前目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT_PATH = os.path.join(PROJECT_ROOT, "data", "financial_eval", "sft", "sft_react.jsonl")
QA_PATH = os.path.join(PROJECT_ROOT, "data", "financial_eval", "train_qa_pairs_zh.json")
OUTPUT_TRAIN = os.path.join(PROJECT_ROOT, "data", "financial_eval", "grpo_agentic_train.parquet")
OUTPUT_VAL = os.path.join(PROJECT_ROOT, "data", "financial_eval", "grpo_agentic_val.parquet")
VAL_RATIO = 0.1


def _build_aliases_index(qa_path):
    """从 QA 数据构建 question → 评分信息索引。

    不能以 gold answer 作为键：公司名、人名等答案在多道题中会重复，
    会把不同问题的 gold chunks 相互覆盖。
    """
    with open(qa_path) as f:
        qa_data = json.load(f)
    index = {}
    for item in qa_data:
        gold = item.get("final_answer", "")
        aliases = item.get("answer_aliases", [])
        chunks = [hop.get("doc_chunk_id", "") for hop in item.get("hops", []) if hop.get("doc_chunk_id")]
        info = {"gold": gold, "aliases": aliases, "gold_chunks": chunks}
        questions = {
            item.get("final_question", "").strip(),
            item.get("final_question_zh", "").strip(),
            item.get("final_question_en", "").strip(),
        }
        for question in questions - {""}:
            if question in index and index[question] != info:
                raise ValueError(f"Duplicate question maps to different QA records: {question[:80]}")
            index[question] = info
    print(f"[agentic-grpo] Loaded {len(qa_data)} QA items ({len(index)} question variants)")
    return index


def build_dataset(sft_path=SFT_PATH, qa_path=QA_PATH):
    """从 SFT 数据提取 question/gold，关联 QA 数据的 answer_aliases 和 gold_chunks"""
    qa_index = _build_aliases_index(qa_path)

    records = []
    matched = 0
    localized_gold = 0
    with open(sft_path) as f:
        for i, line in enumerate(f):
            entry = json.loads(line)
            question = entry["question"]
            gold = entry["gold"]
            subset = entry.get("subset", "unknown")
            hop_count = int(entry.get("hop_count", 1))

            # 按问题精确关联，避免重复答案导致 gold chunks 串样。
            qa_info = qa_index.get(question)
            if qa_info is None:
                raise ValueError(f"SFT question not found in QA data (line {i + 1}): {question[:100]}")
            qa_gold = qa_info.get("gold", "")
            if gold != qa_gold:
                # 中文 SFT 中部分答案经过本地化，而 QA 的 canonical answer
                # 仍为英文。这不是错配；保留 SFT gold 作为 target，并把 QA
                # gold 加入可接受别名。
                localized_gold += 1
            aliases = list(qa_info.get("aliases", []))
            if qa_gold and qa_gold != gold and qa_gold not in aliases:
                aliases.append(qa_gold)
            gold_chunks = qa_info.get("gold_chunks", [])
            if not gold_chunks:
                raise ValueError(f"No gold chunks for SFT question at line {i + 1}")
            if aliases:
                matched += 1
            if gold not in aliases:
                aliases = [gold] + aliases

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]

            record = {
                "data_source": "financial_agentic_rag",
                "prompt": prompt,
                "ability": "multi_hop_qa",
                "reward_model": {
                    "ground_truth": {
                        "target": gold,
                        "answer": gold,
                        "question": question,
                        "answer_aliases": aliases,
                        "gold_chunks": gold_chunks,
                        "hop_count": hop_count,
                    },
                },
                "extra_info": {
                    "index": i,
                    "need_tools_kwargs": True,
                    "question": question,
                    "split": "train",
                    "subset": subset,
                    "hop_count": hop_count,
                    "tools_kwargs": {
                        tool_name: {
                            "create_kwargs": {
                                "ground_truth": gold,
                                "question": question,
                                "data_source": "financial_agentic_rag",
                            }
                        }
                        for tool_name in [
                            "keyword_search",
                            "semantic_search",
                            "graph_search",
                            "hybrid_search",
                        ]
                    },
                },
                "metadata": {
                    "subset": subset,
                    "hop_count": hop_count,
                },
            }
            records.append(record)

    print(
        f"[agentic-grpo] {len(records)} records, {matched} with aliases, "
        f"{localized_gold} localized gold answers"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", default=SFT_PATH)
    parser.add_argument("--qa", default=QA_PATH)
    parser.add_argument("--train-output", default=OUTPUT_TRAIN)
    parser.add_argument("--val-output", default=OUTPUT_VAL)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    args = parser.parse_args()
    if not 0 <= args.val_ratio < 1:
        parser.error("--val-ratio must be in [0, 1)")

    print("[agentic-grpo] 生成 Agentic GRPO 训练数据...")

    records = build_dataset(args.sft, args.qa)
    if not records:
        raise ValueError("No training records were generated")

    # train/val split
    import random
    random.seed(42)
    indices = list(range(len(records)))
    random.shuffle(indices)
    val_size = int(len(records) * args.val_ratio)
    val_indices = set(indices[:val_size])

    train_records = [records[i] for i in range(len(records)) if i not in val_indices]
    val_records = [records[i] for i in range(len(records)) if i in val_indices]

    # 验证格式
    sample = records[0]
    assert isinstance(sample["prompt"], list), "prompt 应为 list"
    assert sample["prompt"][0]["role"] == "system", "第一条应为 system"
    assert sample["extra_info"]["need_tools_kwargs"] is True, "need_tools_kwargs 应为 True"

    df_train = pd.DataFrame(train_records)
    df_val = pd.DataFrame(val_records)
    os.makedirs(os.path.dirname(os.path.abspath(args.train_output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.val_output)), exist_ok=True)
    df_train.to_parquet(args.train_output, index=False)
    df_val.to_parquet(args.val_output, index=False)
    print(f"[agentic-grpo] train: {len(df_train)} 条 → {args.train_output}")
    print(f"[agentic-grpo] val:   {len(df_val)} 条 → {args.val_output}")


if __name__ == "__main__":
    main()
