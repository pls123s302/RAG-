from langgraph.graph import StateGraph, START, END

from app.nodes.answer import direct_answer_node, rag_answer_node
from app.nodes.retrieve import retrieve_node
from app.nodes.router import route_question
from app.state import RAGState


def build_graph():
    """
    构建 LangGraph 流程。

    当前流程：
    START
      └── route_question
            ├── rag    -> retrieve -> rag_answer -> END
            └── direct -> direct_answer -> END
    """

    graph_builder = StateGraph(RAGState)

    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("rag_answer", rag_answer_node)
    graph_builder.add_node("direct_answer", direct_answer_node)

    graph_builder.add_conditional_edges(
        START,
        route_question,
        {
            "rag": "retrieve",
            "direct": "direct_answer",
        },
    )

    graph_builder.add_edge("retrieve", "rag_answer")
    graph_builder.add_edge("rag_answer", END)
    graph_builder.add_edge("direct_answer", END)

    return graph_builder.compile()
