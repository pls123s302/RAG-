from langchain_core.documents import Document

from app.models import get_llm
from app.prompts import (
    build_direct_answer_prompt,
    build_rag_answer_prompt,
)
from app.state import RAGState


def format_docs(docs: list[Document]) -> str:
    """
    将检索到的 Document 列表格式化成 Prompt 上下文。

    当前支持两类资料：
    1. 普通文档片段：来自 Chroma
    2. 知识图谱资料：来自 Neo4j
    """

    if not docs:
        return "没有检索到相关资料。"

    formatted_parts = []

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")

        if source == "neo4j_knowledge_graph":
            title = f"【知识图谱资料 {index}】"
        elif source == "neo4j_kg_error":
            title = f"【知识图谱错误 {index}】"
        elif source == "chroma_error":
            title = f"【向量检索错误 {index}】"
        else:
            title = f"【原始文档片段 {index}】"

        metadata_text = ""

        if metadata:
            metadata_items = []

            for key, value in metadata.items():
                metadata_items.append(f"{key}: {value}")

            metadata_text = "\n".join(metadata_items)

        part = f"""
{title}

来源信息：
{metadata_text if metadata_text else "无"}

内容：
{doc.page_content}
""".strip()

        formatted_parts.append(part)

    return "\n\n---\n\n".join(formatted_parts)


def rag_answer_node(state: RAGState) -> RAGState:
    """
    RAG 回答节点。

    使用：
    1. Chroma 检索到的原始文档片段；
    2. Neo4j 检索到的知识图谱关系；
    共同生成答案。
    """

    question = state["question"]
    docs = state.get("docs", [])

    context = format_docs(docs)

    prompt = build_rag_answer_prompt(
        question=question,
        context=context,
    )

    response = get_llm().invoke(prompt)

    return {
        **state,
        "answer": response.content,
    }


def direct_answer_node(state: RAGState) -> RAGState:
    """
    直接回答节点。

    不查询知识库，直接让大模型回答。
    """

    question = state["question"]

    prompt = build_direct_answer_prompt(question)

    response = get_llm().invoke(prompt)

    return {
        **state,
        "answer": response.content,
    }