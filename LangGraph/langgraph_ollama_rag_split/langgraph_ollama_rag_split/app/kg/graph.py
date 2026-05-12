from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.kg.nodes import (
    extract_kg_node,
    prepare_next_doc_node,
    record_result_node,
    should_continue,
    write_kg_node,
)
from app.kg.state import KGBuildState


def has_current_doc(state: KGBuildState) -> str:
    """
    判断当前是否还有需要处理的 Document。

    prepare_next_doc_node 会根据 current_index 取当前 chunk。
    如果 current_doc 是 None，说明已经没有 chunk 需要处理了。
    """

    if state.get("current_doc") is None:
        return "end"

    return "continue"


def build_kg_graph():
    """
    构建知识图谱入库 LangGraph。

    流程：

    START
      ↓
    prepare_next_doc
      ↓
    是否还有 current_doc？
      ├── 没有 -> END
      └── 有   -> extract_kg
                    ↓
                  write_kg
                    ↓
                  record_result
                    ↓
                  是否还有下一个 chunk？
                    ├── 有   -> prepare_next_doc
                    └── 没有 -> END
    """

    graph = StateGraph(KGBuildState)

    graph.add_node("prepare_next_doc", prepare_next_doc_node)
    graph.add_node("extract_kg", extract_kg_node)
    graph.add_node("write_kg", write_kg_node)
    graph.add_node("record_result", record_result_node)

    graph.add_edge(START, "prepare_next_doc")

    graph.add_conditional_edges(
        "prepare_next_doc",
        has_current_doc,
        {
            "continue": "extract_kg",
            "end": END,
        },
    )

    graph.add_edge("extract_kg", "write_kg")
    graph.add_edge("write_kg", "record_result")

    graph.add_conditional_edges(
        "record_result",
        should_continue,
        {
            "continue": "prepare_next_doc",
            "end": END,
        },
    )

    return graph.compile()


def build_knowledge_graph_with_langgraph(
    docs: list[Document],
) -> dict[str, Any]:
    """
    对外暴露的知识图谱构建入口。

    以后只需要调用这个函数，就可以把一批 Document chunk
    通过 LangGraph 流程抽取实体关系并写入 Neo4j。
    """

    if not docs:
        return {
            "docs": [],
            "current_index": 0,
            "current_doc": None,
            "kg_data": {
                "entities": [],
                "relationships": [],
            },
            "results": [],
            "error": None,
        }

    app = build_kg_graph()

    initial_state: KGBuildState = {
        "docs": docs,
        "current_index": 0,
        "current_doc": None,
        "kg_data": {
            "entities": [],
            "relationships": [],
        },
        "results": [],
        "error": None,
    }

    final_state = app.invoke(initial_state)

    return final_state