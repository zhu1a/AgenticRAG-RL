"""基础评测指标：EM、F1、CostTracker"""
import re
import unicodedata
from collections import Counter

#answer_aliases 是答案的同义写法白名单，用来让 EM/F1 对“中英文、单位、格式、缩写”更公平；它是 LLM 生成的，不是语义理解，也不能替代 LLM Judge。
#aliases 只能加分、不能减分

def _normalize(text: str) -> str:
    """标准化文本用于评测比较"""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    # 同时移除中英文标点。
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    # 移除冠词
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # 合并空白
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    """中英文混合评测分词。

    英文/数字保留为词，中文按字计算 F1。这避免无空格的中文答案
    被当成一个整体 token，使 F1 退化成 EM。
    """
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]", _normalize(text))


def exact_match(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """EM 精确匹配，支持 answer_aliases"""
    candidates = [gold] + (aliases or [])
    return max(1.0 if _normalize(prediction) == _normalize(c) else 0.0 for c in candidates)


def f1_score(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    """Token-level F1，支持 answer_aliases（取最大值）"""
    candidates = [gold] + (aliases or [])
    return max(_f1_single(prediction, c) for c in candidates)

#按token级对比
def _f1_single(prediction: str, gold: str) -> float:
    """单个 gold 的 Token-level F1"""
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class CostTracker:
    """统计 LLM 调用次数、工具调用次数、延迟"""

    def __init__(self):
        self.records = []

    def record(self, state: dict, latency: float):
        self.records.append({
            "total_tool_calls": state.get("total_tool_calls", 0),
            "iteration_count": state.get("iteration_count", 0),
            "latency": latency,
        })

    def summary(self) -> dict:
        if not self.records:
            return {}
        n = len(self.records)
        return {
            "num_queries": n,
            "avg_tool_calls": sum(r["total_tool_calls"] for r in self.records) / n,
            "avg_iterations": sum(r["iteration_count"] for r in self.records) / n,
            "avg_latency": sum(r["latency"] for r in self.records) / n,
            "total_latency": sum(r["latency"] for r in self.records),
        }
