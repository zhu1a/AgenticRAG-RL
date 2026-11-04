# AgenticRAG + RL

面向金融多跳问答的 Agentic RAG 项目，覆盖混合检索、LangGraph Agent 编排、
Oracle 轨迹构造、LoRA SFT、verl GRPO 训练与评测。

完整说明、环境配置和复现步骤见 [core/README.md](core/README.md)。

## 主要模块

- `core/agents/`：Router、Planner、Executor、Verifier、Synthesizer
- `core/retrieval/`：BM25、FAISS、知识图谱、RRF 融合与重排
- `core/scripts/`：索引、数据合成、轨迹转换与评测脚本
- `core/training/`：SFT、GRPO、Reward 和检索工具服务
- `core/data/`：金融语料、QA、SFT 与 GRPO 数据
- `core/evaluation/`：EM、F1、Hop Recall 与 LLM Judge

## 仓库说明

模型权重、训练输出、运行日志及可由语料重新构建的二进制检索索引
（FAISS、BM25、chunk store、entity embeddings）未提交到 Git。
请按照 `core/README.md` 下载模型并运行索引构建脚本。

`core/LLaMA-Factory/` 和 `core/verl/` 为项目训练流程使用的框架代码副本，
其许可与上游项目保持一致。
