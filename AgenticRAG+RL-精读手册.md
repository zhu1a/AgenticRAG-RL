# AgenticRAG+RL 精读手册

> 目标：读完之后，你能**画出两条链路、说清每个数字的来源、判断哪些结论可信**，并且知道要改哪里。
>
> 约定：以下路径默认相对 `core/`；所有数字来自对当前仓库文件的直接统计与交叉核对（不是照抄 README）。

---

## 0. 先记住三句话

1. **任务**：金融多跳问答（2–4 跳，比较/推理），语料是 7 份 2025 年半年报切成的 1933 个 chunk，答案是短实体/数字/结论。
2. **仓库里有两套 Agent，不要混着看**：
   - **A 套（Prompt 编排）**：`agents/graph.py` 的 LangGraph `router → planner → executor → verifier → synthesizer`，不训练，靠 prompt 驱动任意 LLM。
   - **B 套（模型自己会调工具）**：SFT + GRPO 训练一个 Qwen3-4B，让它自己输出 `<tool_call>`，环境执行检索，最多 7 轮后 `<answer>`。
   - 两套共用语料/索引/QA，但**检索代码是两份、评测入口是两套、指标不可直接横比**。
3. **训练链路**：合成数据 → 划分 → oracle 轨迹 → ReAct 格式 SFT（冷启动）→ GRPO（用检索命中率 + Judge 做奖励）。

---

## 1. 项目到底在做什么

### 1.1 任务与难点

输入一句中文问题，例如：

> 与上年同期相比，哪个变化幅度更大：步步高 2025 半年报的经营活动现金净流量，还是红旗连锁的营业成本？

输出一个短答案（`经营活动现金流净额` / `-19.94%` / `是`）。难点不在生成，而在：

- **证据分散**：答案的两个比较项在两份不同公司的报告里，甚至在不同页的表格里；
- **多跳**：第 2 跳的检索词往往依赖第 1 跳的答案（“它”“该公司”“这个科目”）；
- **答案形式短**：需要别名匹配，否则 EM 会把“对的意思”判错；
- **检索噪声**：财报表格被切成 500 字 chunk 后，数字和表头经常分离。

### 1.2 两条技术路线（全文最重要的一张表）

| 维度 | A 套：LangGraph 编排 | B 套：训练出来的 Agent |
|---|---|---|
| 入口 | `agents/graph.py:27` `build_graph()` | verl `tool_agent` 循环 / `scripts/eval_agentic.py` |
| 谁决定下一步 | planner / verifier 两个 LLM 节点 | 同一个模型在对话里生成 `<tool_call>` |
| 工具 | 4 个 Python 函数（keyword/semantic/graph/read_chunk） | 4 个 verl `BaseTool`（keyword/semantic/graph/hybrid） |
| 循环 | PEV：验证不足就重规划，最多 3/5 轮、10/15 次工具调用 | 最多 7 个 assistant turn |
| 是否训练 | 否 | LoRA SFT + GRPO |
| 评测入口 | `evaluation/run_eval.py`、`scripts/run_cloud_eval.py` | `scripts/eval_agentic.py` |
| 主要指标 | EM/F1、hop_recall、Judge 三维度、成本 | EM/F1、轮数、工具调用数、按 hop 分组 |
| 典型用途 | 强基线 / 云端大模型 / 消融 | 论文主方法：小模型端到端 |

> 读代码时先问自己：“我现在看的是 A 还是 B？” 90% 的困惑来自把两套的类名当成同一套。

### 1.3 仓库边界

- 自研代码约 **8.4k 行**（`agents/`、`retrieval/`、`evaluation/`、`training/`、`scripts/`）。
- `LLaMA-Factory/` 与 `verl/` 是**完整 vendored 框架**，不要从头读；只需要看它们与本项目的 3 个接缝：
  1. `training/sft_zh_react.yaml` + `LLaMA-Factory/data/dataset_info.json`；
  2. `training/config/financial_tool_config.yaml`（verl 工具注册）；
  3. `training/reward_agentic_rag.py` 的 `compute_score` 签名。
- README 提到的 `docs/project_report_v2.md` **不存在**；`results/`、`logs/`、`training/outputs/` 也都不在仓库里。所以仓库内**没有可核验的实验结果**，只有脚本注释里记录的历史数字。

---

## 2. 全局地图

### 2.1 目录职责

| 目录 | 真正的作用 | 先读哪个文件 |
|---|---|---|
| `config.py` | 全局路径/模型/检索参数，全部可用环境变量覆盖 | 全文 42 行，先读 |
| `agents/` | A 套 LangGraph 编排 | `state.py` → `graph.py` → `executor.py` |
| `retrieval/` | 索引与 4 种检索工具 | `semantic_search.py` → `hybrid_search.py` → `graph_search.py` |
| `llm/` | 统一 OpenAI 协议客户端（本地 vLLM + 云端） | `client.py` |
| `evaluation/` | A 套评测 + Judge + 诊断 + 消融 | `metrics.py` → `run_eval.py` |
| `training/` | SFT 配置、GRPO 奖励、verl 工具、启动脚本 | `reward_agentic_rag.py` |
| `scripts/` | 数据合成、格式转换、B 套评测、工具脚本 | 见第 3、7、9 章 |
| `data/` | 语料、索引、QA、SFT/GRPO 数据 | 先看 `financial_all/` 与 `financial_eval/` |

### 2.2 数据流（从 PDF 到训练样本）

```
7 份 2025 半年报 PDF
  └─ parse_pdf_corpus.py（按段落切 ~500 字，带 pages/section）
       └─ data/financial_all/corpus_all.json        1933 chunks
            ├─ build_index.py
            │    ├─ faiss.index        1933 × 1024（BGE-M3，归一化，内积=余弦）
            │    ├─ bm25.pkl           jieba 分词后的 BM25Okapi
            │    ├─ chunk_store.pkl    chunk_id → {text,title,pages,section}
            │    └─ chunk_ids.json     FAISS 行号 → chunk_id
            └─ build_knowledge_graph.py
                 ├─ knowledge_graph.json    16146 节点 / 16634 边
                 └─ entity_embeddings.pkl   16146 × 1024

gen_seed_qa.py（每 chunk 生成 3 个原子 QA）
  └─ domain_multihop_synthesis.py（逐跳扩展 + 四重验证）
       └─ 1340 条多跳 QA
            ├─ split_train_test.py → 1065 train / 275 test
            │    ├─ 1065 → clean_synthesis.py → 797 clean
            │    │    ├─ SFT：1065（未清洗）→ traces_oracle → sft_react.jsonl
            │    │    └─ GRPO：797 → 718 train / 79 val
            │    └─ 275 → 翻译/清洗 → 185（qa_pairs_zh_clean.json，最终中文评测集）
            └─ annotate_search_tools.py → 给每个 hop 标注哪些工具能召回 gold chunk
```

> 这几个数字不是拍脑袋：`split_train_test.py` 把 train 重编号为 `mhop_000000…`、test 重编号为 `mhop_test_000000…`；现有文件是 1065 + 275，id 规则完全吻合。而 185 的 id **全部落在 275 的 id 空间里**（`mhop_test_000000…000274`），且同一 id 的问题在 275 里是英文、在 185 里是中文——所以 185 是 275 的**中文翻译/清洗子集**，不是另一次独立划分。

### 2.3 A 套运行时序

```
query
  │
  ├─ router（1 次 LLM，JSON）── simple ─→ simple_rag（semantic top3）─→ END
  │
  └─ multi_hop
       └─ planner（1 次 LLM，JSON 计划）
            └─ executor（循环执行 plan 中可执行的步骤，每次工具调用）
                 └─ verifier（1 次 LLM，判断 sufficient / insufficient）
                      ├─ sufficient ──────────────→ synthesizer（1 次 LLM）→ END
                      └─ insufficient 且未超预算 ─→ planner（iteration+1）
```

单条 query 的 LLM 调用数约为：`1(router) + 1(planner)×轮数 + 1(verifier)×轮数 + 1(synthesizer)`，再加工具调用。这是后面算成本的地方。

### 2.4 B 套运行时序

```
system(含 # Tools) + user question
  → 模型输出 "<think>…</think><tool_call>{name,arguments}</tool_call>"
  → 环境执行工具，返回 "<tool_response>…</tool_response>"
  → 模型继续 …（≤7 个 assistant turn）
  → 模型输出 "<answer>…</answer>"
```

没有 planner/verifier 节点；**“规划”和“验证”都隐式发生在模型自己的思维链里**。这是 A/B 两套最本质的区别。

---

## 3. 数据工程精读

### 3.1 语料：7 份报告、1933 个 chunk

`data/financial_all/corpus_all.json` 的实际构成：

| 公司 | chunk 前缀 | chunk 数 |
|---|---|---:|
| 名创优品 | `mcyp` | 425 |
| 永辉超市 | `yh` | 377 |
| 家家悦 | `jjy` | 343 |
| 步步高 | `bbg` | 255 |
| 中百控股 | `zb` | 244 |
| 成都红旗连锁 | `hq` | 180 |
| 高鑫零售 | `gx` | 109 |

平均 chunk 长度 **487 字**，平均覆盖 **1.48 页**。每个 chunk 带 `title`、`pages`、`section`，`pages` 对后面“排除同页相邻 chunk”的去重很关键。

> 注意：`scripts/parse_pdf_corpus.py` 生成的是 `fin_0000` 这种 id，而实际语料是 `yh_0000` 这种公司前缀。说明**真正生成 `corpus_all.json` 的版本没有完整进仓库**（或在别处后处理过）。复现语料时不能直接拿这个脚本对齐现有索引。

### 3.2 种子 QA → 多跳合成（`domain_multihop_synthesis.py`）

这是数据侧最值得读的 1200 行。核心循环：

1. **种子**：每个 chunk 用 LLM 生成 3 个原子 QA（答案 ≤10 token、答案不能出现在问题里、不能含 and/or…）。
2. **逐跳扩展** `process_seed()`（`:360`）：
   - 当前已有 `hop_1…hop_k`；
   - 用**上一跳的 final_question** 做主查询、用**上一跳的 refined_answer** 做辅助查询，双路检索；
   - 再跑一路 `graph_search`，三路结果按 `chunk_id` 合并；
   - 排除已用过的 chunk，并排除**同公司、页码 ±1** 的相邻 chunk（防止把同一张表切成两半当两跳）；
   - 对新 chunk 生成原子 QA → 精炼答案 → 调 LLM 把“已有 k 跳 + 新 1 跳”合并成一个新的多跳问题（`inference` 或 `comparison`）。
3. **四重验证**（`:845-968`），全过才收：
   - **语义检查**：LLM 判断新问题是否真的需要两段文档、答案是否一致；
   - **推理检查**：不给文档，让 LLM 直接答；**答对反而失败**（说明不需要检索）；
   - **单文档检查**：只给部分文档，LLM 应该答不出；
   - **全文档检查**：给全部文档，LLM 应该答对。
4. **配额**：每层 hop 有输出配额（默认 `2:1,3:3,4:3`），comparison 不超过 40%，每个 seed 最多 5 条。

这套“**去掉一篇就答不出、给全就能答出**”的验证，是多跳数据质量的核心思想，比单纯让 LLM 打分可靠得多。

### 3.3 数据血缘与关键数字（务必记住）

| 数据集 | 条数 | 用途 | 备注 |
|---|---:|---|---|
| `data/datasets/qa_pairs.json` | 1305 | AgenticRAGTracer 原始英文通用域 | 1511 chunks，2–4 跳 |
| `data/financial_eval/train_qa_pairs_zh.json` | 1065 | 金融训练全集 | `final_question` 竟是**英文**，中文在 `final_question_zh` |
| `data/financial_eval/train_qa_pairs_zh_clean.json` | 797 | 清洗后训练集 | `final_question` 是中文 |
| `data/financial_eval/qa_pairs_zh_clean.json` | 185 | **最终中文评测集** | id 为 `mhop_test_*`，是 275 的子集 |
| `data/financial_eval/qa_pairs.json` | 275 | 完整 test（英文为主） | 185 条是它的中文子集 |
| `data/financial_eval/sft/sft_react.jsonl` | 1065 | SFT | 由 1065 条 oracle trace 生成 |
| `.../grpo_agentic_train.parquet` | 718 | GRPO 训练 | 来自 797 clean 的 90% |
| `.../grpo_agentic_val.parquet` | 79 | GRPO 验证 | 来自 797 clean 的 10% |

评测集 185 的分布：2 跳 130 / 3 跳 45 / 4 跳 10；`2hop_comparison` 102、`3hop_inference` 38、`2hop_inference` 28、`3hop_comparison` 7、`4hop_comparison` 6、`4hop_inference` 4。

两个容易踩的点：

- **同名字段语言不一致**：`train_qa_pairs_zh.json` 的 `final_question` 是英文，`train_qa_pairs_zh_clean.json` 的 `final_question` 是中文。直接 `qa["final_question"]` 喂给中文 Agent 时，必须先确认自己在哪个文件上。
- **训练/评测同文档不同题**：185 与 1065 的 question 零重叠，但 gold chunk 有 164/276 重叠（同一批年报）。这不算题目泄漏，但意味着**评测集和训练集共享实体/表格**，分数会偏乐观。

### 3.4 工具标注：一份“有信息但没用上”的数据

`annotate_search_tools.py` 对每个非首跳，实际跑 3 个单工具 + 4 种 hybrid 组合，检查 gold chunk 是否在 top-10 里，把命中的工具写进 `search_tools`。

以 185 评测集为例（250 个非首跳）：

| 命中工具组合 | 数量 |
|---|---:|
| keyword + semantic | 76 |
| graph + keyword + semantic | 65 |
| semantic only | 48 |
| graph + semantic | 27 |
| graph only | 14 |
| keyword only | 12 |
| graph + keyword | 8 |

这本来可以用来教模型“**哪一步该用哪个工具**”。但实际生成 SFT 数据时用的是**未标注**的 `train_qa_pairs_zh.json`（`search_tools` 全空），`build_oracle_traces.py:45-48` 于是回退成 `keyword_search`。结果：**1065 条 SFT 样本里的 2570 次工具调用，100% 是 keyword_search**。

---

## 4. 检索层精读

### 4.1 索引文件（`data/financial_all/indexes/`）

| 文件 | 内容 | 谁在用 |
|---|---|---|
| `faiss.index` | `IndexFlatIP`，1933×1024，向量已归一化 | `semantic_search.py`、B 套 SemanticTool |
| `bm25.pkl` | `BM25Okapi`，jieba 分词 | `keyword_search.py`、B 套 KeywordTool |
| `chunk_store.pkl` | `chunk_id → {text,title,pages,section}` | 所有工具 |
| `chunk_ids.json` | FAISS 行号 → chunk_id | FAISS 结果反查 |
| `knowledge_graph.json` | node-link 格式，16146 节点/16634 边 | `graph_search.py`、B 套 GraphTool |
| `entity_embeddings.pkl` | `{entities, entity_to_idx, embeddings}`，16146×1024 | 图谱实体匹配 |

`data/indexes/` 是另一套：**1511** 个 chunk，对应 AgenticRAGTracer 原始英文语料。`config.py:32-37` 默认指向它，所以不做环境变量覆盖就会“拿英文索引跑中文问题”。

### 4.2 四种工具

| 工具 | 召回 | 精排 | 适用 |
|---|---|---|---|
| `keyword_search` | BM25 top-20 | 若 >5 条再 rerank 到 5 | 精确名称、科目名、数字 |
| `semantic_search` | FAISS top-20 | rerank 到 5 | 概念性、描述性查询 |
| `graph_search` | 实体 embedding top-5 → 2 跳 BFS → 20 候选 | rerank 到 5 | 跨文档实体关系、比较 |
| `read_chunk` | 按 id 直读 | 无 | 已知 chunk_id 时补全上下文 |
| `hybrid_search`（B 套） | 多工具并行 → RRF | rerank 到 3 | 高召回关键步骤 |

参数在 `config.py:27-29`：`RERANK_TOP_K=5`、`SEMANTIC_TOP_K=20`、`BM25_TOP_K=20`。

### 4.3 一次 `semantic_search` 的完整调用链

以 A 套为例：

```
executor.execute_step()                      agents/executor.py:100
  → semantic_search(query)                   retrieval/semantic_search.py:28
      → _load()                              :18   懒加载 faiss/chunk_ids/chunk_store
      → embedder.encode([query])             retrieval/embedder.py:26
           BGE-M3，normalize_embeddings=True
      → index.search(q_vec, 20)              :35
      → rerank(query, 20 个 passage, top_k=5) retrieval/reranker.py:25
           BGE-reranker-v2-m3，CrossEncoder 打分
      → 返回 [{chunk_id,text,title,score,source}]
```

两个细节：

- `embedder._get_model(device=None)` 默认 **CPU**（`embedder.py:14-16`）。`run_eval.py:77-101` 用 monkeypatch 把默认设备改成 GPU；但 `api/server.py` 直接调 `run_query` 时没有这层 patch，会默默跑在 CPU 上。
- reranker 的输入是 `[query, passage]` 对，20 个候选就是 20 次 cross-encoder 前向；hybrid 只取 15 个候选，是成本控制。

### 4.4 RRF + rerank（`retrieval/hybrid_search.py`）

```python
score(chunk) = Σ_over_lists 1 / (60 + rank + 1)
```

- 按 `chunk_id` 去重，累加排名分；**完全不用原始分数**，所以 BM25 的 12.3 和 FAISS 的 0.71 可以直接融合。
- 取融合后的前 15 个候选做 cross-encoder 重排，再取 top-5。
- `k=60` 是 RRF 论文的常用平滑值：排名越靠后，分差越小，避免某一路的“第 1 名”压倒性支配。

代价：每次 hybrid 要并行跑 2–3 路检索 + 15 对 rerank。对 4B 模型的 7 轮 rollout 来说，这是主要延迟来源之一。

### 4.5 图谱检索（`retrieval/graph_search.py`）

```
query → BGE-M3 编码
  → 与 16146 个实体向量点积，取 top-5 种子实体（无阈值）
  → MultiDiGraph 上 BFS 2 跳（同时走出边和入边）
  → 收集节点 mentions 里的 chunk + 边上的 chunk_id
  → 分数 = 种子相似度 × 1/(1+depth)
  → 取 20 个候选 → rerank → top-5
```

图是用 LLM 从每个 chunk 抽 3–8 个三元组建的（`build_knowledge_graph.py:24-58`），关系如“数量为”“位于”“营业收入为”。它的价值在**实体桥接**：query 里出现“永辉”，图能带到“永辉 → 董事 → 某某”这类语义检索不一定排得高的 chunk。

它的局限也很明显：

- 实体匹配是**稠密向量最近邻**，没有别名/缩写归一化；
- BFS 不看关系类型，2 跳容易扩散；
- A 套没设相似度阈值，永远取 5 个种子，可能引入噪声；
- **B 套和 `eval_agentic.py` 里读错了节点属性**（见第 11 章 P0-2），实际返回空。

---

## 5. Agent 编排层（A 套）

### 5.1 State：哪些字段累积，哪些字段覆盖

`agents/state.py:6-18`：

```python
query, query_type, plan, current_step, verification_result,
verification_feedback, final_answer, iteration_count, total_tool_calls
evidence:   Annotated[list, operator.add]   # 累积
tool_calls: Annotated[list, operator.add]   # 累积
trace:      Annotated[list, operator.add]   # 累积
```

关键理解：LangGraph 对**带 reducer 的字段做合并**，不带的字段**整体替换**。所以：

- `evidence/trace` 每轮追加，不会丢历史；
- `plan/current_step` 每次 planner 重新赋值，天然支持“重规划”；
- `iteration_count` 在 planner 里 `+1`（`planner.py:58`），所以它统计的是“规划了几轮”，不是“执行了几轮”。

### 5.2 五个节点逐个读

**router**（`agents/router.py:10-23`）
- 一次 LLM JSON 调用，输出 `{"query_type": "simple"|"multi_hop"}`。
- 任何异常/解析失败都**默认 multi_hop**（保守策略：宁可多查）。
- 代价：每条 query 多一次 LLM 调用；收益：简单题不必走完整 PEV。

**planner**（`agents/planner.py:17-60`）
- 输入：query、可用工具列表、可选的上一轮 verifier 反馈 + 已试过的检索摘要。
- 输出：`[{"id", "sub_query", "tool", "depends_on"}]`，全部标 `pending`。
- 失败回退：单步 `semantic_search`（`:48-49`），保证图不会断。
- `tool` 可以是字符串，也可以是列表（表示多工具融合，A 套交给 `multi_tool_search`）。

**executor**（`agents/executor.py:43-134`）
- 一个节点内部用 `while` 连续执行**所有依赖已满足的步骤**，直到计划跑完或 `total_tool_calls >= max_calls`。
- 依赖检查：`depends_on` 里的 id 必须已出现在 evidence 的 `step_id` 中（`:60-63`）。
- 多工具：`len(tool_names) > 1` 或 `hybrid_search` → 并行调用 + RRF（`:83-96`）。
- **上下文增强**：第 2 步起，把上一步第一条结果的 `text[:200]` 拼到子查询后面（`:76-80`）。这能帮助指代消解，但也可能把 BM25 带偏。
- 每条 evidence 只保留 `results[:5]`（`:110`）。

**verifier**（`agents/verifier.py:24-67`）
- 把 evidence 的每条前 3 个结果、每个 `text[:500]` 拼成证据文本，让 LLM 判 `sufficient/insufficient`。
- small profile 的 prompt 明确要求“**宽松**”：只要关键实体出现过就判 sufficient（`prompts.py:69`）。这是为了防止小模型无限重规划。
- 有一个“证据去重”保护（`:46-61`）：如果重规划后的新证据与旧证据 chunk 重叠 ≥90%，强制 sufficient。
- `after_verification`（`:70-82`）：`sufficient` 或预算耗尽 → synthesizer；否则 → planner。

**synthesizer**（`agents/synthesizer.py:39-57`）
- 证据只取每条前 3 个、每个 `text[:500]`；prompt 要求“只输出答案本身”。
- `_extract_short_answer`（`:10-36`）：优先 `<answer>` 标签，去前缀，取第一行，超长取第一句。

**simple_rag**（`synthesizer.py:60-85`）
- 单次 `semantic_search`，取 top-3 拼上下文，直接答。
- 它也会写一条 evidence，所以评测里 hop_recall 仍能算（但只有 1 次检索）。

### 5.3 循环、预算与终止

| 预算 | small | large |
|---|---:|---:|
| `max_iterations`（规划轮数） | 3 | 5 |
| `max_retrieval_calls`（工具调用数） | 10 | 15 |

两个预算分别在 `executor.py:56`（节点内）和 `verifier.py:77-78`（条件边）检查。终止条件只有两个：**verifier 说 sufficient** 或 **预算耗尽**。

> 注意：`get_profile()` 的正则只匹配 7B/8B/14B（`prompts.py:335`）。所以 **Qwen3-4B 命中的是 large profile**（5 轮/15 次），不是 small。如果作者本意是“小模型用小预算”，这是一个隐藏的错配。

### 5.4 Prompt 分级

`prompts.py` 用 `(模型大小, 语言)` 组合出 4 个 profile：

- `small` / `large`：planner 的“工具选择自由度”、verifier 的严格程度、预算不同；
- `_zh` 后缀：全部中文模板；
- 选择逻辑在 `get_profile()`（`:338-349`），语言读 `config.PROMPT_LANG`。

一个隐蔽点：`prompts.py:3` 在**导入时**把 `AGENT_LLM_MODEL` 绑成了模块常量。`run_eval.py:60-63` 运行时改的是 `config.AGENT_LLM_MODEL`，**不会**影响 `get_profile()` 的模型名判断。所以 `--model` 换模型时，prompt profile 可能还是按启动时的环境变量选的。

### 5.5 用一条真实 trace 走一遍

`data/financial_eval/traces_oracle_zh.jsonl` 第一条（oracle 轨迹，不是模型跑出来的，但结构完全一致）：

```
Q: 永辉“非同一控制企业合并资产评估增值”的递延所得税负债期末余额，
   与中百“固定资产一次性扣除”的递延所得税负债期末余额，哪个更大？

router: multi_hop
planner:
  1. “非同一控制企业合并资产评估增值”的递延所得税负债期末余额是多少？
     tool=keyword_search, depends_on=[]
  2. 固定资产一次性扣除的递延所得税负债期末余额是多少？
     tool=[keyword_search, semantic_search, graph_search], depends_on=[1]

executor:
  step1 → yh_0252（永辉报表附注，含 12,594,757.20）
  step2 → zb_0185（中百报表附注，含 60,347,007.86）

verifier: sufficient
synthesizer: 固定资产一次性扣除
```

这条 trace 同时展示了三件事：**计划是 DAG**（第 2 步依赖第 1 步）、**工具可以是多路**、**evidence 里带 chunk_id 和原文**。SFT 数据就是把这个结构翻译成 ReAct 对话。

---

## 6. LLM 接入层（`llm/client.py`）

设计很干净：**所有模型都走 OpenAI 兼容协议**，本地 vLLM 与云端 API 没有代码分叉。

- `ModelConfig`（`:44-65`）：url / model_name / api_key / 采样参数 / 重试次数；`client` 懒加载。
- `MODEL_CONFIGS`（`:76-93`）：内置 `Qwen3-4B`、`Qwen3-32B`、`gpt-oss-120b`；新增模型只需加一条。
- `get_from_llm`（`:154-211`）：最多重试 20 次，指数退避；空回复也重试；`_strip_think` 去掉 think 段。
- `agent_chat` / `judge_chat`：分别用 `AGENT_LLM_MODEL` / `JUDGE_LLM_MODEL`，并记录到线程安全的 `LLMStats`。
- `_extract_json`（`:217-235`）：支持 ```json 代码块、`{...}`、`[...]` 三种提取，解析失败返回 None。
- `agent_chat_json`（`:252-259`）：JSON 解析失败会**重试 2 次**（每次都重新调模型）。

两个工程注意点：

1. `extra_body` 里传了 `top_k`、`min_p`、`chat_template_kwargs`（`:193-197`），这些是 **vLLM 扩展参数**；某些云端 OpenAI 兼容端点会直接 400。接云端时要么去掉，要么确认对方支持。
2. `max_tokens=config.max_len`（`:181`），而 `Qwen3-32B` 的 `max_len=131072`。也就是说默认可能请求 13 万 token 的输出上限，成本和延迟都不可控；实际模型会提前停，但账单/超时风险在。

---

## 7. SFT：教格式，不教检索

### 7.1 Oracle trace：不跑 Agent，直接“抄答案”

`build_oracle_traces.py` 不调用模型，而是把 QA 的 `hops` 直接翻译成轨迹：

- 每个 hop 的 `question` → `sub_query`；
- `hop.doc_chunk_id` → 从 corpus 取原文（截 500 字）作为 evidence；
- `search_tools` 为空 → 默认 `keyword_search`；
- `verifier` 永远 `sufficient`，`em=1.0`。

好处：零推理成本、标签 100% 正确、可大规模生成。
代价：**模型看到的是“每次检索都命中唯一正确 chunk”的世界**。

> 文件名有坑：SFT 实际用的是 `traces_oracle.jsonl`（1065 条，内容是中文），它由 `train_qa_pairs_zh.json` 的 `final_question_zh` 生成；`traces_oracle_zh.jsonl` 反而是 797 条 clean 训练集的轨迹。看到 `_zh` 不要想当然。

### 7.2 ReAct 格式（`trace_to_sft.py:106-156`）

每一步生成两条消息：

```json
{"role":"assistant","content":"<think>I need to search for: …</think>\n<tool_call>\n{\"name\":\"keyword_search\",\"arguments\":{\"query\":\"…\"}}\n</tool_call>"}
{"role":"tool","content":"[yh_0252] 23,095,220.64 …"}
```

最后一条：

```json
{"role":"assistant","content":"<answer>固定资产一次性扣除</answer>"}
```

多工具步骤会被折叠成 `hybrid_search(query, tools=[...])`（`_normalize_tool`，`:71-92`）。

### 7.3 与 LLaMA-Factory 的接缝

`convert_sft_to_llamafactory.py:37-79` 做了一件很讲究的事：

1. 用 **Qwen3 的 tokenizer** `apply_chat_template(msgs, tools=TOOL_SCHEMAS)` 生成完整训练文本；
2. 再按 `<|im_start|>role` 切回 `messages`，写成 sharegpt；
3. 这样 SFT 的 tokenization 与 GRPO rollout 的 hermes 格式**尽量一致**。

LLaMA-Factory 侧（`sft_zh_react.yaml`）：

| 参数 | 值 |
|---|---|
| 基座 | Qwen3-4B |
| 方法 | LoRA，`rank=64`，`target=all` |
| 数据 | `financial_agent_zh_react`（1065 条） |
| 模板 | `qwen3_nothink` |
| cutoff | 2048 |
| batch | 2 × 8 = 16 |
| lr | 1e-4，cosine，warmup 0.1 |
| epoch | 3 |
| 精度 | fp16 |

### 7.4 这一步真正教了什么

| 教了 | 没教 |
|---|---|
| `<think>/<tool_call>/<tool_response>/<answer>` 格式 | 工具选择（2570 次全是 keyword） |
| “一跳一次检索、拿到结果再答”的节奏 | 多工具融合（hybrid 只在 schema 里出现） |
| 从一段财报文本里抽短答案 | 抗噪（observation 永远是 gold chunk，无 top-k 噪声） |
| 中文金融术语的表述 | 检索失败后的重试/换词（oracle 从不失败） |

所以 SFT 的定位是**冷启动格式与领域语感**，不是“教会 Agent 检索”。真正学检索策略的是 GRPO。

---

## 8. GRPO：奖励塑形是核心

### 8.1 verl 多轮 Agent 配置（`start_grpo.sh`）

| 项 | 值 | 含义 |
|---|---|---|
| 算法 | `adv_estimator=grpo` | 组内归一化优势，不需要 critic |
| Agent loop | `tool_agent` | 多轮工具调用 |
| 格式 | `hermes` | `<tool_call>` / `<tool_response>` |
| 最大 assistant turn | 7 | 与评测一致 |
| rollout n | 4 | 每个 prompt 采 4 条轨迹 |
| train batch | 32 | 全局 batch；ppo mini 16 |
| prompt/response | 1024 / 1024 | 长问题会被 `filter_overlong_prompts` 丢掉 |
| max_model_len | 4096 | vLLM |
| KL | 0.05，`low_var_kl` | 防止跑偏 |
| lr | 5e-6 | 比 SFT 小 20 倍 |
| epoch | 2 | 脚本注释却写“1 epoch” |
| GPU | 4 卡，vLLM mem 0.35 | 训练与 rollout 混部 |

工具通过 `training/config/financial_tool_config.yaml` 注册，索引目录指向 `data/financial_all/indexes`；embedding/reranker 不在训练进程里，而是通过 `training/tools/retrieval_server.py` 起在**另一张卡（GPU 7）**上，用 HTTP 调用（约 2GB 显存）。这是一个很实用的工程解耦：**训练卡不跑 BGE**。

### 8.2 奖励函数逐项拆解（`training/reward_agentic_rag.py`）

`compute_score(solution_str, ground_truth)` 的总分：

```
score = hop_pr×0.30 + faith×0.25 + corr×0.25 + grounded×0.10 + format×0.10 - insufficient_penalty
score = clamp(score, 0, 1)
```

| 项 | 权重 | 计算方式 | 想强化的行为 | 实际状态 |
|---|---:|---|---|---|
| `hop_pr` | 0.30 | 从文本里抽 `[chunk_id]`，与 `gold_chunks` 算 F1；为 0 时给 **-0.05** | 检索到 gold chunk，且别捞太多噪声 | 依赖 gold_chunks 正确性；26% 训练样本的 gold_chunks 是错的（见 P0-3） |
| `faith` | 0.25 | LLM Judge：答案是否被 evidence 支持 | 答案有据可依 | **实际恒为 0**：evidence 提取正则失效（P0-1） |
| `corr` | 0.25 | 若 EM=1 或 F1≥0.8 → 1；否则 LLM Judge 0/0.3/0.5/0.7/1 | 答对 | 正常，但 Judge 失败会静默给 0 |
| `grounded` | 0.10 | 答案 token 与 evidence token 的重叠率 | 答案词出现在证据里 | **实际恒为 0**（同一个 evidence 提取问题） |
| `format` | 0.10 | `<answer>` 0.06 + `<tool_call>` 0.04 | 遵守输出格式 | 正常 |
| `insufficient` | -0.05 | 工具调用数 < hop_count | 至少查够跳数 | 正常 |

另外两个硬规则：

- `</think>` 出现 >5 次 → 直接 0（防复读）；
- 无答案时只给 `format + hop - penalty`，并 `max(0, …)`。

**这意味着按仓库里这份 reward，实际可拿到的上限约 0.65**（faith 0.25 + grounded 0.10 永远拿不到）。脚本注释里的“Faith×0.25”与代码行为不一致。

### 8.3 三阶段课程（脚本注释里的故事）

`start_grpo.sh` 头部记录了作者的调参路径：

| 阶段 | 版本 | 奖励侧重 | 目标 |
|---|---|---|---|
| Stage 1 | v11e | Faith×0.30 | 先学会 grounding |
| Stage 2 | v14e | Corr×0.40 | 强化 correctness |
| Stage 3 | v15e | hop_pr×0.30 | 提升 CtxP（检索精度） |

Stage 3 的 base 是 `qwen3-4b-grpo-v14e-step30-merged`，注释给出 `Judge_C=0.334, Faith=0.199, CtxP=0.251`。但：

- 这个 base 模型**不在仓库里**（`training/outputs/` 不存在）；
- `start_grpo.sh:38` 指向 `training/reward_agentic_rag_v9a.py`，**该文件不存在**，仓库里只有 `reward_agentic_rag.py`；
- 注释说“1 epoch”，实际 `total_epochs=2`。

所以这条训练链**不能按原样复现**，只能按代码逻辑复刻。

### 8.4 奖励里的工程细节

- `hop_pr` 的 `-0.05` 是有意设计：如果只给 `0.30×0` 就没有梯度信号，负值能推动模型“至少别一条都没中”。
- `corr` 先用 EM/F1 做便宜判断，只有 F1<0.8 才调 Judge，省 API。
- `corr` 和 `faith` 用 `ThreadPoolExecutor(max_workers=2)` 并行（`:275-279`），因为 Judge 是主要延迟。
- `_call_judge` 失败不抛异常，返回 `{}` → 分数 0。训练时如果 Judge 服务抖动，会悄悄变成“答错”信号。

---

## 9. 评测：三套入口，三套口径

### 9.1 三个入口的分工

| 入口 | 评谁 | 指标 | 关键特点 |
|---|---|---|---|
| `evaluation/run_eval.py` | A 套 LangGraph | EM/F1 + diagnostics + 可选 Judge | 多线程、checkpoint/resume、GPU 预热 |
| `scripts/run_cloud_eval.py` | A 套（云端模型） | EM/F1 + 按 subset | 扁平并行、用 aliases、工具集**不含 graph** |
| `scripts/eval_agentic.py` | B 套训练模型 | EM/F1 + turns/tools + 按 hop | 独立实现检索、原生 `<tool_call>` 解析 |
| `scripts/run_llm_judge.py` | 离线补 Judge | correctness/faithfulness/ctx_precision | 但数据路径与结果 schema 对不上（见 P1-2） |
| `evaluation/ablation.py` | A 套消融 | EM/F1/hop_recall | 5 个配置，可关 verifier/工具 |

### 9.2 指标：EM / F1 / hop_recall

`evaluation/metrics.py`：

- `_normalize`：小写、去 **ASCII** 标点、去冠词、压空白；
- `exact_match`：`pred == gold`（可选 aliases）；
- `f1_score`：token-level F1（可选 aliases）；
- `CostTracker`：平均工具调用数、平均迭代数、平均延迟。

`evaluation/hop_aware_eval.py`：

- `hop_recall = |gold_chunks ∩ retrieved_chunks| / |gold_chunks|`；
- `premature_collapse`：hop_recall<0.5 且 iteration≤1（证据不足就停了）；
- `over_extension`：工具调用 > 3×hop_count；
- `step_alignment`：evidence 条数 / hop_count；
- 支持按 2/3/4 hop 分组。

### 9.3 LLM Judge（`evaluation/llm_judge.py`）

三个维度，各 5 档（1.0/0.7/0.5/0.3/0.0）：

| 维度 | 输入 | 判断 |
|---|---|---|
| Correctness | question, pred, gold | 语义是否等价 |
| Faithfulness | question, answer, evidence | 答案的每个 claim 是否被证据支持（**不判断对错**） |
| Context Precision | question, evidence, gold docs | 检索文档覆盖了多少 gold 文档的信息 |

输入截断：evidence 6000 字符、gold docs 3000 字符。中文/英文各一套 prompt。

### 9.4 评测口径的三个不一致

1. **aliases**：`run_eval.py:157-158` 调 `exact_match/f1_score` 时**没传** `answer_aliases`；`run_cloud_eval.py:64-65` 和 `eval_agentic.py` 的 F1 传了。185 条评测集平均 **21.4 个别名/条**，不传会显著低估。
2. **中文标点**：`metrics._normalize` 只去掉 `string.punctuation`（ASCII），中文的“。”“，”不去；`eval_agentic.py` 的 `_normalize` 做了全角→半角、中文标点、数字/汉字加空格。
3. **EM 是否用 aliases**：`eval_agentic.py:400` 的 EM 只比 gold，F1 才用 aliases；`run_cloud_eval` 两者都用 aliases。

结论：**不要把 `run_eval` 的 EM/F1 和 `run_cloud_eval`/`eval_agentic` 的表直接放一起**，除非先统一口径重算。

### 9.5 消融配置（`evaluation/ablation.py:13-39`）

| 配置 | 工具 | verifier | 想回答的问题 |
|---|---|---|---|
| `full_system` | keyword, semantic, read_chunk | 是 | 完整系统（**注意：不含 graph**） |
| `no_keyword` | semantic, read_chunk | 是 | BM25 有用吗 |
| `no_read_chunk` | keyword, semantic | 是 | 读原文有用吗 |
| `no_verifier` | keyword, semantic, read_chunk | 否 | 自纠错循环有用吗 |
| `semantic_only` | semantic | 否 | Naive RAG 基线 |

`full_system` 不含 `graph_search`，所以它不是 `build_graph()` 的默认全量工具集；“full”这个名字有误导性。

---

## 10. 关键设计取舍（面试/答辩可直接用）

| 决策 | 收益 | 代价 / 风险 |
|---|---|---|
| **oracle trace 代替真实 rollout** | 零推理成本、标签干净、可批量 | 模型没见过检索失败；observation 是 gold chunk，存在 exposure bias |
| **SFT 全部用 keyword_search** | 格式统一、训练稳定 | 没教工具选择；与 GRPO 的四工具/噪声检索有分布差 |
| **RRF 融合 + cross-encoder 重排** | 对 BM25/FAISS 分数尺度不敏感；精度高 | 每次多路检索 + 15 对 rerank，延迟高 |
| **Planner/Verifier 双 LLM 循环** | 自适应计算；能重规划 | 每轮多 2 次 LLM 调用；verifier 判错会早停/空转 |
| **hop_pr 作为稠密奖励** | 直接强化“检索到 gold” | 依赖 gold_chunks 正确；chunk_id 必须出现在 rollout 文本里 |
| **Judge 进 reward** | 能奖励语义正确而非字面匹配 | 成本、方差、服务故障会静默变 0 |
| **A/B 两套栈并存** | A 可接任意大模型做基线；B 是小模型主方法 | 检索代码重复，已经产生 drift（graph 属性 bug） |
| **中文评测 + 21 个别名** | 更公平地评中文短答案 | 指标实现不统一，跨脚本不可比 |
| **4 卡混部 + 独立 retrieval server** | 训练卡不跑 BGE，显存可控 | 多一个 HTTP 单点；server 挂了工具全空 |

---

## 11. 代码级问题清单（精读的真正价值）

> 以下不是“风格问题”，而是**会改变结果**的问题。按影响排序。

### P0-1 奖励函数的 evidence 提取正则失效 → Faith/Grounded 恒为 0

`training/reward_agentic_rag.py:149-151`：

```python
def _extract_evidence(text: str) -> str:
    matches = re.findall(r'เพิ่มเติม(.*?)', text, re.DOTALL)
    return "\n".join(m.strip() for m in matches if m.strip())
```

`เพิ่มเติม` 是泰文（“更多”），正常的 rollout 文本里不会出现；而且 `(.*?)` 后面没有结束锚点，即使出现了也只会匹配空串。于是：

- `matches` 为空 → `evidence = ""`；
- `faith_score = 0.0`（`:270-282`）；
- `grounded = 0.0`（`:253-255`）。

**影响**：0.25 + 0.10 = **35% 的奖励权重永远拿不到**；模型学不到 grounding。注释与代码不一致。

**修法**：改成实际格式

```python
matches = re.findall(r'<tool_response>(.*?)</tool_response>', text, re.DOTALL)
```

并加一条单测：给一段含 `<tool_response>` 的文本，断言 `_extract_evidence` 非空。

**快速验证**：

```python
import re
re.findall(r'เพิ่มเติม(.*?)', '<tool_response>abc</tool_response>', re.DOTALL)  # [] → evidence = ""
```

### P0-2 图检索的节点属性读错 → B 套的 graph_search 恒为空

建图时节点属性叫 `mentions`（`scripts/build_knowledge_graph.py:148-155`）：

```python
G.add_node(head, mentions=[])
G.nodes[head]["mentions"].append(chunk_id)
```

A 套读对了（`retrieval/graph_search.py:97-98`）：

```python
for cid in node_data.get("mentions", []):
```

但 B 套和 agentic 评测读的是 `chunk_ids`：

- `training/tools/financial_search_tool.py:278`：`node_data.get("chunk_ids", [])`
- `scripts/eval_agentic.py:147`：`graph.nodes[node].get("chunk_ids", [])`

而实际 16146 个节点里 **0 个**有 `chunk_ids`，全部只有 `mentions`。

**影响**：
- GRPO 的 `graph_search` 永远返回 `(no results)`；
- `hybrid_search` 里的 graph 一路贡献 0，RRF 实际只有 keyword+semantic；
- `eval_agentic` 的图检索同样为空（但 A 套 `run_eval` 不受影响）。

**修法**：两处改成 `mentions`；再加一个索引自检脚本，断言至少一个节点能取到 chunk。

### P0-3 GRPO 数据的 gold_chunks / aliases 串号 → 26% 样本奖励错误

`scripts/prepare_agentic_grpo_data.py:30-43` 用 **gold answer 字符串**做索引：

```python
index[gold] = {"aliases": aliases, "gold_chunks": chunks}
```

但很多问题的最终答案相同（“步步高商业连锁股份有限公司”出现 25 次，“家家悦”21 次，“永辉”16 次……）。后面覆盖前面，**所有同答案的题都拿到最后一条的 chunks/aliases**。

实测（与 QA 源文件逐题对比）：

| 数据 | 行数 | gold_chunks 错 | 比例 |
|---|---:|---:|---:|
| `grpo_agentic_train.parquet` | 718 | 187 | **26.0%** |
| `grpo_agentic_val.parquet` | 79 | 15 | 19.0% |
| aliases 不一致（train） | 718 | 207 | 28.8% |

例：问“步步高 vs 家家悦 哪家净利润增幅更高”，parquet 给的 chunks 却是 `('bbg_0184','jjy_0252')`——另一道题的证据。

**影响**：`hop_pr`（0.30）对约 1/4 样本给出错误目标；模型检索到正确 chunk 反而可能不得分。这是**比 reward 正则更隐蔽的数据污染**。

**修法**：用 `(question, gold)` 或 `qa_id` 做 key；或直接按 question 遍历原 QA 文件，不要建 answer 索引。

### P1-1 多个被引用文件不存在，脚本无法按原样跑

| 引用处 | 缺失文件 |
|---|---|
| `training/start_grpo.sh:38` | `training/reward_agentic_rag_v9a.py` |
| `training/sft_pipeline.sh:17` | `training/qwen3_4b_sft_zh_func_call.yaml` |
| `training/sft_pipeline.sh:226` | `scripts/run_judge_batch.py` |
| README / 文档 | `docs/project_report_v2.md` |
| `sft_pipeline.sh:23` | `data/financial_eval_zh/`（评测数据目录） |

另外 `LLaMA-Factory/data/dataset_info.json` 里 `financial_agent_zh_react` 的 `file_name` 是**绝对路径** `/share/project/kouqian/...`，换机器直接失效。

### P1-2 评测数据路径与结果 schema 对不上

- `run_eval.py:14-21` 读 `<data_dir>/qa_pairs.json`；但最终中文评测集叫 `qa_pairs_zh_clean.json`（185），而 `data/financial_eval/qa_pairs.json` 是 275 条完整 test（英文为主，185 是其翻译子集）。`sft_pipeline.sh` 又指向不存在的 `data/financial_eval_zh`。
- `run_llm_judge.py:59-70` 假设结果是 `{config_name: {"per_sample": [...]}}`（消融格式）；而 `run_eval` 输出是 `{"summary":…, "results":[…]}`。直接跑会在 `results["results"].get(...)` 处报错。
- `run_llm_judge.py:50-56` 固定读 `data/datasets/qa_pairs.json` + `corpus.json`（英文 AgenticRAGTracer），无法给中文金融题构造 gold docs。

**结果**：README 宣称的“三维度 Judge”在仓库现有脚本下**无法对 185 条中文评测集复现**。

### P1-3 指标口径不统一（见 9.4）

同一份预测，`run_eval` 和 `run_cloud_eval` 可能差几个点。做对比实验前先统一：aliases、中文标点、全角半角、是否 strip 句号。

### P1-4 SFT 与 GRPO 的分布差

| 维度 | SFT | GRPO |
|---|---|---|
| 工具 schema | 4（keyword/semantic/graph/hybrid） | 4（同一套） |
| 实际用到的工具 | 100% keyword | 4 选 1 / hybrid |
| system / tools | 同一句 system，模板注入 `# Tools` | 相同 |
| observation | 1 条 gold chunk | top-3 检索结果（含噪声） |
| 失败恢复 | 无 | 可重试 7 轮 |
| 问题语言 | 中文 | 中文 |

这不是“bug”，但意味着 GRPO 一上来就要同时学“换工具”和“抗噪”，冷启动难度被放大。

### P2 级问题（不影响主结论，但值得知道）

1. **verifier 的去重保护基本失效**：`evidence` 只记 `step_id`，重规划后新 plan 的 id 又从 1 开始，`current_step_ids` 与旧 evidence 的 id 重合，`prev_evidence` 恒为空（`verifier.py:50-61`）。防循环实际只靠预算。
2. **executor 依赖死锁**：若 `depends_on` 引用了永远不会完成的 id，`while` 会 break，条件边又返回 `execute`，节点间空转直到 LangGraph 的 recursion limit（`executor.py:60-63`、`:137-146`）。
3. **embedder 默认 CPU**：`embedder.py:14-16` 的 `device=None → "cpu"`；A 套 `semantic_search` 不传 device，`api/server.py` 会跑 CPU。`run_eval` 用 monkeypatch 规避。
4. **prompt profile 的模型名是导入期常量**：`prompts.py:3` + `:344-346`，运行时 `--model` 不改变 size 选择。
5. **4B 命中 large profile**：正则只认 7/8/14B；Qwen3-4B 用 5 轮/15 次预算。
6. **语料 id 前缀对不上**：`parse_pdf_corpus.py` 产出 `fin_XXXX`，实际是 `yh_XXXX`；真正的语料构建脚本未入库。
7. **`eval_agentic` 的 hybrid RRF 不是真 RRF**：它把多工具结果拼成一个列表后按全局 rank 算 `1/(60+rank+1)`（`eval_agentic.py:180-187`），没有按“每路各自的 rank”融合。
8. **`max_tokens=max_len`**：`llm/client.py:181`，32B 配置下可能请求 131k 输出。
9. **`total_epochs=2` 与注释“1 epoch”不一致**；`start_grpo.sh` 的 base 模型路径也不存在。
10. **`ablation.full_system` 不含 graph_search**，与 `build_graph()` 默认工具集不一致。

### 一键自检脚本（纯标准库，不需要装依赖）

```python
# 在 core/ 下运行
import json, re
G = json.load(open('data/financial_all/indexes/knowledge_graph.json'))
print('nodes with chunk_ids:', sum('chunk_ids' in n for n in G['nodes']))   # 应为 0 → 证实 P0-2
print('nodes with mentions :', sum('mentions' in n for n in G['nodes']))    # 16146

print('thai regex:', re.findall(r'เพิ่มเติม(.*?)', '<tool_response>abc</tool_response>', re.DOTALL))  # [] → P0-1

import pandas as pd
qa = json.load(open('data/financial_eval/train_qa_pairs_zh_clean.json'))
q2c = {x['final_question']: sorted(h['doc_chunk_id'] for h in x['hops'] if h.get('doc_chunk_id')) for x in qa}
df = pd.read_parquet('data/financial_eval/grpo_agentic_train.parquet')
bad = sum(1 for _, r in df.iterrows()
          if sorted(map(str, r['reward_model']['ground_truth']['gold_chunks'])) != q2c.get(r['extra_info']['question']))
print('gold_chunks mismatch:', bad, '/', len(df))   # 187/718 → P0-3
```

---

## 12. 复现路线（按“能跑”排序）

### 12.0 先认清环境现状

- 仓库里**没有** `models/`（BGE-M3、reranker 需自行下载）。
- 当前机器未安装 `langgraph/faiss/jieba/rank_bm25/sentence_transformers/openai`，无法直接跑。
- `LLaMA-Factory` 与 `verl` 是完整源码，但需要各自的 CUDA/PyTorch 环境，且**必须隔离**（README 明确两个 conda env）。

### 12.1 只读路线（无 GPU，先建立直觉）

```bash
cd core
python3 - <<'PY'
import json
from collections import Counter
qa = json.load(open('data/financial_eval/qa_pairs_zh_clean.json'))
print(len(qa), Counter(q['subset'] for q in qa))
print(json.dumps(qa[0], ensure_ascii=False, indent=2)[:1500])
PY
```

再读 `traces_oracle_zh.jsonl` 第一条，对照 5.5 节走一遍 plan/evidence/answer。

### 12.2 A 套最小评测闭环（需要 GPU + vLLM）

```bash
# 1) 环境
conda create -n agenticrag python=3.11
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2) 模型
mkdir -p models
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3

# 3) 把 185 条整理成评测入口期望的文件名
mkdir -p data/financial_eval_zh
cp data/financial_eval/qa_pairs_zh_clean.json data/financial_eval_zh/qa_pairs.json

# 4) 起 vLLM
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B --served-model-name Qwen3-4B \
  --port 9097 --gpu-memory-utilization 0.45 --max-model-len 32768

# 5) 跑 A 套
export VLLM_BASE_URL=http://localhost:9097/v1
export AGENT_LLM_MODEL=Qwen3-4B
export PROMPT_LANG=zh
export NEWS_CORPUS_DIR=data/financial_eval
export NEWS_INDEX_DIR=data/financial_all/indexes
python -u evaluation/run_eval.py \
  --model Qwen3-4B --workers 4 --max-samples 185 \
  --data-dir data/financial_eval_zh \
  --index-dir data/financial_all/indexes --lang zh
```

> 注意：`run_eval` 不读 aliases，得到的 EM 会偏低；要公平比较，后续用 `run_cloud_eval` 或自己补 aliases。

### 12.3 B 套评测闭环

```bash
export RETRIEVAL_DEVICE=cuda:1
python -u scripts/eval_agentic.py \
  --model Qwen3-4B-SFT-ZH \
  --port 9097 \
  --data-dir data/financial_eval_zh \
  --index-dir data/financial_all/indexes \
  --max-samples 185 --max-turns 7 --workers 4
```

前提：`eval_agentic` 自己加载 BGE（`models/bge-m3`、`models/bge-reranker-v2-m3`）。图检索目前是空的（P0-2），修好后再看 graph 的贡献。

### 12.4 SFT

```bash
# 1) 修 dataset_info 的绝对路径，指向本仓库的 lf_react.jsonl
# 2) 修 sft_pipeline.sh 的 YAML_PATH（指向存在的 training/sft_zh_react.yaml）
export LLAMA_FACTORY=/path/to/core/LLaMA-Factory
bash training/sft_pipeline.sh 3 0 1
```

注意 `sft_pipeline.sh` 仍会调用缺失的 `run_judge_batch.py`，需自己补或注释掉 Step 5。

### 12.5 GRPO

```bash
# 1) 修 start_grpo.sh:38 的 REWARD_FN，指向 training/reward_agentic_rag.py
# 2) 起 retrieval server（单独一张卡）
bash training/start_retrieval_server.sh

# 3) 确认 tool_config 的 index_dir 指向 data/financial_all/indexes
# 4) 起训练
export VERL_DIR=/path/to/core/verl
bash training/start_grpo.sh
```

训练前建议先做 12.6 的三个修复，否则奖励信号是残的。

### 12.6 如果只能改三处

1. **P0-1**：修 `_extract_evidence` → 恢复 35% 奖励权重；
2. **P0-3**：重建 GRPO parquet 的 gold_chunks → 去掉 26% 的错误监督；
3. **P0-2**：修 graph 节点属性 → 让第 4 个工具真正参与。

这三处都不改模型结构，只改数据和奖励，是性价比最高的实验起点。

---

## 13. 阅读路线与自测

### 13.1 分阶段路线

**第 1 遍（30 分钟）——建立骨架**
`README` → `config.py` → `agents/state.py` → `agents/graph.py` → `agents/executor.py` → `agents/synthesizer.py`。
目标：能画出 2.3 的流程图，说清 evidence/trace 为什么用 `operator.add`。

**第 2 遍（2 小时）——补检索与评测**
`retrieval/semantic_search.py` → `keyword_search.py` → `hybrid_search.py` → `graph_search.py` → `evaluation/metrics.py` → `evaluation/run_eval.py` → `llm_judge.py`。
目标：能说清一次 semantic_search 的 6 个步骤、RRF 为什么不用原始分数、EM 为什么会低估中文。

**第 3 遍（半天）——数据与 SFT**
`domain_multihop_synthesis.py` 的 `process_seed` / `_extend_one_hop` / `_verify_and_build` → `build_oracle_traces.py` → `trace_to_sft.py` → `convert_sft_to_llamafactory.py` → `sft_zh_react.yaml`。
目标：能解释四重验证、oracle trace 的三个偏差、SFT 为什么全是 keyword。

**第 4 遍（一天）——GRPO 与实验设计**
`reward_agentic_rag.py` → `start_grpo.sh` → `financial_tool_config.yaml` → `financial_search_tool.py` → `verl` 的 `tool_agent`（只看接缝）。
目标：能逐项解释奖励、指出 P0-1/P0-3、设计一个消融。

**可以跳过**：`LLaMA-Factory/src/**`、`verl/verl/trainer/**` 的内部实现；先把它们当黑盒，只读配置和调用点。

### 13.2 自测题（能答上来就算读懂）

1. 为什么 `evidence/trace` 用 `operator.add`，而 `plan/current_step` 不用？
   <details>前者要跨轮累积；后者每轮 planner 重新生成，必须整体替换。</details>

2. executor 里已经有 `while` 了，为什么还需要 `should_continue_executing` 条件边？
   <details>while 负责“一次节点调用内尽量多跑几步”；条件边负责“节点之间是否还要再进来”，处理依赖阻塞和预算耗尽。</details>

3. RRF 的 `k=60` 和 `1/(k+rank+1)` 各起什么作用？为什么融合后还要 rerank？
   <details>k 平滑排名分差，避免头部压倒性；rank 越小分越高。RRF 只利用排名、丢失相关性强度，cross-encoder 用 query-passage 交互补精度。</details>

4. verifier 的“去重保护”为什么基本不生效？
   <details>evidence 只存 step_id，重规划后 plan 的 id 又从 1 开始，旧 evidence 全被归为“当前轮”，prev 集合为空。</details>

5. oracle trace 有哪三个分布偏差？
   <details>(a) observation 是唯一 gold chunk，无检索噪声；(b) 工具全是 keyword，没教选择；(c) verifier 永远 sufficient，没有失败恢复。</details>

6. GRPO 奖励里哪几项实际失效？为什么？
   <details>faith 0.25 与 grounded 0.10；`_extract_evidence` 用泰文正则，永远提取不到 `<tool_response>` 内容。</details>

7. 为什么 1065 条 SFT 样本里 2570 次工具调用全是 keyword_search？
   <details>trace 用未标注的 `train_qa_pairs_zh.json` 生成，hop 的 `search_tools` 为空，`build_oracle_traces` 回退到 keyword_search。</details>

8. `run_eval` 的 EM 为什么可能明显低于 `run_cloud_eval`？
   <details>前者不传 answer_aliases，且 `_normalize` 不去中文标点/全角；后者两者都处理。</details>

9. 为什么 A 套和 B 套的分数不能直接横比？
   <details>prompt/工具/循环/评测脚本/指标口径都不同；A 是 prompt 编排，B 是训练模型；甚至检索实现是两份。</details>

10. 如果只改一处提升训练效果，你改哪里？为什么？
    <details>优先修 P0-1（恢复 35% 奖励）或 P0-3（去掉 26% 错误 gold）；前者改 3 行代码，后者改数据构建的 key。两者都比调超参收益更确定。</details>

### 13.3 90 秒口述版

> 这个项目做金融多跳 QA：7 份半年报、1933 个 chunk，问题 2–4 跳，答案很短。仓库里有两条链路：一条是 LangGraph 的 planner-executor-verifier 编排，不训练，用来接大模型做基线；另一条是 Qwen3-4B 的 SFT+GRPO，模型自己输出 tool_call，环境执行四种检索，最多 7 轮。数据侧先用种子 QA 逐跳扩展，再用“去掉一篇答不出、给全就能答出”的四重验证筛出 1340 条，划成 1065 训练 / 275 测试；测试集翻译清洗成 185 条中文评测，训练集清洗成 797 条给 GRPO。SFT 用 oracle 轨迹教格式，但只教了 keyword_search，而且 observation 是 gold chunk；GRPO 的奖励是 hop 命中率 0.30、Judge 正确性 0.25、忠实度 0.25、grounded 0.10、格式 0.10。我读下来最关键的三个问题是：奖励里的 evidence 正则写错导致忠实度恒为 0；GRPO 数据按答案字符串建索引，26% 的 gold chunk 串号；图检索在训练侧读错节点属性，实际一直返回空。

---

## 附录 A：文件速查

| 文件 | 一句话 |
|---|---|
| `config.py` | 所有路径/模型/检索参数，环境变量可覆盖 |
| `agents/state.py` | LangGraph 状态定义（哪些字段累积） |
| `agents/graph.py` | 图组装：router/simple_rag/planner/executor/verifier/synthesizer |
| `agents/router.py` | simple vs multi_hop 分类 |
| `agents/planner.py` | 查询分解为 DAG，支持重规划反馈 |
| `agents/executor.py` | 按计划调工具，多工具走 RRF |
| `agents/verifier.py` | 证据充分性判断 + 预算终止 |
| `agents/synthesizer.py` | 生成短答案；simple_rag 直通 |
| `agents/prompts.py` | 中英 × 大小模型 4 套 prompt + 预算 |
| `retrieval/embedder.py` | BGE-M3 封装（多设备、线程安全） |
| `retrieval/reranker.py` | BGE-reranker-v2-m3 封装 |
| `retrieval/semantic_search.py` | FAISS top-20 → rerank top-5 |
| `retrieval/keyword_search.py` | jieba + BM25 top-20 → rerank |
| `retrieval/graph_search.py` | 实体匹配 → 2 跳 BFS → rerank |
| `retrieval/hybrid_search.py` | RRF 融合 + rerank |
| `retrieval/read_chunk.py` | 按 chunk_id 读原文 |
| `llm/client.py` | 统一 OpenAI 协议 + 重试 + JSON 提取 |
| `evaluation/metrics.py` | EM / F1 / 成本统计 |
| `evaluation/hop_aware_eval.py` | hop_recall / 早停 / 过度检索诊断 |
| `evaluation/llm_judge.py` | 三维度 Judge prompt |
| `evaluation/run_eval.py` | A 套评测主入口（checkpoint/resume） |
| `evaluation/ablation.py` | 5 个消融配置 |
| `scripts/run_cloud_eval.py` | A 套扁平并行评测（用 aliases） |
| `scripts/eval_agentic.py` | B 套多轮 tool_call 评测 |
| `scripts/run_llm_judge.py` | 离线补 Judge（schema 不兼容） |
| `scripts/domain_multihop_synthesis.py` | 多跳合成 + 四重验证 |
| `scripts/gen_seed_qa.py` | 从 chunk 生成原子 QA |
| `scripts/annotate_search_tools.py` | 标注每个 hop 的工具召回 |
| `scripts/build_oracle_traces.py` | QA → oracle 轨迹 |
| `scripts/trace_to_sft.py` | 轨迹 → ReAct 对话 |
| `scripts/convert_sft_to_llamafactory.py` | 用 Qwen3 tokenizer 对齐后转 sharegpt |
| `scripts/prepare_agentic_grpo_data.py` | SFT → verl parquet（有 P0-3） |
| `scripts/build_index.py` | FAISS + BM25 + chunk store |
| `scripts/build_knowledge_graph.py` | LLM 抽三元组 → NetworkX → 实体向量 |
| `training/reward_agentic_rag.py` | GRPO 奖励 v9a（有 P0-1） |
| `training/start_grpo.sh` | verl 4 卡 GRPO 启动 |
| `training/sft_zh_react.yaml` | Qwen3-4B LoRA SFT |
| `training/tools/financial_search_tool.py` | verl 4 个 BaseTool（有 P0-2） |
| `training/tools/retrieval_server.py` | BGE HTTP 服务（embed/rerank） |
| `training/config/financial_tool_config.yaml` | verl 工具注册 |

## 附录 B：关键数字卡

```
语料       7 份报告 / 1933 chunks / 平均 487 字 / 1.48 页
图谱       16146 节点 / 16634 边 / 实体向量 16146×1024
原始基准   1305 QA / 1511 chunks（英文通用域）
多跳数据   1340 = 1065 train + 275 test；275 → 185 中文评测子集
清洗       1065 → 797
SFT        1065 条；2570 次工具调用，100% keyword；平均 2.41 次/条
GRPO       797 → 718 train / 79 val；平均 2.31 个 gold chunk
评测       185 条：2 跳 130 / 3 跳 45 / 4 跳 10
别名       平均 21.4 个/条
奖励       hop 0.30 + faith 0.25 + corr 0.25 + grounded 0.10 + format 0.10 - 0.05
实际上限   约 0.65（faith/grounded 失效）
```

## 附录 C：术语

| 术语 | 含义 |
|---|---|
| chunk | 财报切块，约 500 字，带 `chunk_id/title/pages/section` |
| hop | 一跳 = 一次“找到某个事实”的检索+阅读 |
| oracle trace | 用 ground-truth hops 直接构造的完美轨迹 |
| PEV | Planner → Executor → Verifier 循环 |
| RRF | Reciprocal Rank Fusion，按排名融合多路结果 |
| GRPO | Group Relative Policy Optimization，组内比较算优势 |
| LoRA | 低秩适配器；本项目 rank=64，target=all |
| CtxP | Context Precision，检索文档对 gold 文档的覆盖度 |
| exposure bias | 训练只见正确 observation，推理要面对噪声 |

---

*本文基于当前仓库快照的静态阅读与数据统计；没有运行模型，因此不引用任何未在仓库中出现的实验分数。*
