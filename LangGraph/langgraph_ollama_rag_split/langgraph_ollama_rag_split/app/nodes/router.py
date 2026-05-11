from app.models import get_llm
from app.prompts import build_route_prompt
from app.state import RAGState


def route_question(state: RAGState) -> str:
    """
    路由判断：决定当前问题走 RAG 还是直接回答。
    返回值必须和 graph.py 里的条件边 key 对应。
    """

    question = state["question"]
    prompt = build_route_prompt(question)

    result = get_llm().invoke(prompt).content.strip().upper()

    if "RAG" in result:
        return "rag"
    return "direct"
