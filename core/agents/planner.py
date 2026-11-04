"""规划器：将复杂查询分解为子任务 DAG"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.client import agent_chat_json
from agents.state import AgentState
from agents.prompts import get_profile

TOOL_DESCRIPTIONS = {
    "keyword_search": "BM25 keyword search, good for exact names/entities",
    "semantic_search": "Dense retrieval + reranking, good for semantic similarity",
    "read_chunk": "Read a specific document by chunk_id (use when you have a specific chunk_id from previous steps)",
    "graph_search": "Knowledge graph search: finds related entities and documents through entity relationship traversal. Best for multi-hop questions involving entity connections (e.g., 'Who directed the film starring X?')",
}


def _normalize_plan(raw_plan, query: str) -> list[dict]:
    """把不稳定的 LLM JSON 规范为 Executor 可安全消费的步骤列表。"""
    if not isinstance(raw_plan, list):
        raw_plan = []

    valid_items = [item for item in raw_plan if isinstance(item, dict)]
    if not valid_items:
        valid_items = [{"sub_query": query, "tool": "semantic_search", "depends_on": []}]

    id_map = {}
    for new_id, item in enumerate(valid_items, 1):
        old_id = item.get("id", new_id)
        if old_id not in id_map:
            id_map[old_id] = new_id

    normalized = []
    for new_id, item in enumerate(valid_items, 1):
        sub_query = item.get("sub_query")
        if not isinstance(sub_query, str) or not sub_query.strip():
            sub_query = query

        tool = item.get("tool", "semantic_search")
        if not isinstance(tool, str) or not tool.strip():
            tool = "semantic_search"

        dependencies = item.get("depends_on", [])
        if not isinstance(dependencies, list):
            dependencies = [dependencies] if dependencies not in (None, "") else []
        mapped_dependencies = []
        for dependency in dependencies:
            mapped = id_map.get(dependency)
            if mapped and mapped != new_id and mapped not in mapped_dependencies:
                mapped_dependencies.append(mapped)

        normalized.append({
            "id": new_id,
            "sub_query": sub_query.strip(),
            "tool": tool.strip(),
            "depends_on": mapped_dependencies,
            "status": "pending",
        })
    return normalized


def plan(state: AgentState) -> AgentState:
    """LangGraph node: 生成或重新生成检索计划"""
    query = state["query"]
    iteration = state.get("iteration_count", 0)

    profile = get_profile()

    feedback_section = ""
    if iteration > 0 and state.get("verification_feedback"):
        evidence_summary = "\n".join(
            f"- Step {e['step_id']} [{e.get('tool', '?')}]: \"{e['sub_query']}\" -> {len(e.get('results', []))} results"
            for e in state.get("evidence", [])
        )
        feedback_section = profile["replan_feedback"].format(
            feedback=state["verification_feedback"],
            evidence_summary=evidence_summary or "No evidence yet",
        )

    # 动态生成可用工具列表（消融实验时 TOOL_REGISTRY 可能被过滤）
    from agents.executor import TOOL_REGISTRY, _ensure_tools
    _ensure_tools()
    tools_section = "\n".join(
        f"- {name}: {TOOL_DESCRIPTIONS.get(name, 'search tool')}"
        for name in TOOL_REGISTRY
    )
    if not tools_section:
        tools_section = "- semantic_search: Dense retrieval + reranking"

    prompt = profile["planner"].format(query=query, feedback_section=feedback_section, tools_section=tools_section)
    result = agent_chat_json(prompt)

    result = _normalize_plan(result, query)

    return {
        "plan": result,
        "current_step": 0,
        "iteration_count": iteration + 1,
        "trace": [{"node": "planner", "iteration": iteration + 1, "plan": result}],
    }
