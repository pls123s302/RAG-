from langchain_core.documents import Document

from app.models import get_llm
from app.prompts import build_direct_answer_prompt, build_rag_answer_prompt
from app.state import RAGState


def format_docs(docs: list[Document]) -> str:
    """
    将检索结果格式化为给大模型看的上下文。
    """

    return "\n\n".join(
        [
            f"【资料 {i + 1}】\n来源：{doc.metadata}\n内容：{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )


def rag_answer_node(state: RAGState) -> RAGState:
    """
    RAG 回答节点：基于检索资料回答。
    """

    question = state["question"]
    docs = state["docs"]

    context = format_docs(docs)
    prompt = build_rag_answer_prompt(question, context)

    answer = get_llm().invoke(prompt).content

    return {
        **state,
        "answer": answer,
    }


def direct_answer_node(state: RAGState) -> RAGState:
    """
    直接回答节点：不查知识库。
    """

    question = state["question"]
    prompt = build_direct_answer_prompt(question)

    answer = get_llm().invoke(prompt).content

    return {
        **state,
        "docs": [],
        "route": "direct",
        "answer": answer,
    }
