from langchain_core.documents import Document

from app.config import (
    ENABLE_KNOWLEDGE_GRAPH,
    RETRIEVE_TOP_K,
)
from app.state import RAGState
from app.vectorstore import get_vector_db


def retrieve_node(state: RAGState) -> RAGState:
    """
    RAG 检索节点。

    当前检索逻辑：
    1. 从 Chroma 向量库召回原始文本 chunk；
    2. 如果开启 Neo4j 知识图谱，则额外从 Neo4j 召回实体关系；
    3. 将两类结果合并到 docs 中，交给后续回答节点。
    """

    question = state["question"]

    docs: list[Document] = []

    # 1. Chroma 向量检索
    try:
        vector_db = get_vector_db()

        vector_docs = vector_db.similarity_search(
            question,
            k=RETRIEVE_TOP_K,
        )

        docs.extend(vector_docs)

        print(f"[Retrieve] Chroma 召回 {len(vector_docs)} 条文档。")

    except Exception as exc:
        print(f"[Retrieve] Chroma 检索失败：{exc}")

        docs.append(
            Document(
                page_content=f"Chroma 向量检索失败：{exc}",
                metadata={"source": "chroma_error"},
            )
        )

    # 2. Neo4j 知识图谱检索
    if ENABLE_KNOWLEDGE_GRAPH:
        try:
            from app.kg.retriever import retrieve_graph_context

            graph_docs = retrieve_graph_context(question)

            docs.extend(graph_docs)

            print(f"[Retrieve] Neo4j 知识图谱召回 {len(graph_docs)} 条上下文。")

        except Exception as exc:
            print(f"[Retrieve] Neo4j 知识图谱检索失败：{exc}")

            docs.append(
                Document(
                    page_content=f"Neo4j 知识图谱检索失败：{exc}",
                    metadata={"source": "neo4j_kg_error"},
                )
            )

    return {
        **state,
        "docs": docs,
        "route": "rag",
    }