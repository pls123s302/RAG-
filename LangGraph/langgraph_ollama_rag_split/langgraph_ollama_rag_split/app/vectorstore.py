from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    ENABLE_KNOWLEDGE_GRAPH,
)
from app.models import get_embeddings


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    """
    获取 Chroma 向量库对象。

    persist_directory 表示本地持久化路径。
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )


def add_documents(docs: List[Document]) -> None:
    """
    添加文档到知识库。

    当前流程：
    1. 写入 Chroma 向量库；
    2. 如果开启 ENABLE_KNOWLEDGE_GRAPH，则调用 LangGraph；
    3. LangGraph 负责抽取实体关系并写入 Neo4j。
    """

    if not docs:
        print("没有需要写入的文档。")
        return

    vector_db = get_vector_db()

    # 1. 写入 Chroma
    vector_db.add_documents(docs)

    print(f"[Chroma] 已写入 {len(docs)} 个文档 chunk。")

    # 2. 同步构建 Neo4j 知识图谱
    if ENABLE_KNOWLEDGE_GRAPH:
        try:
            from app.kg.graph import build_knowledge_graph_with_langgraph

            final_state = build_knowledge_graph_with_langgraph(docs)

            results = final_state.get("results", [])

            success_count = sum(1 for item in results if item.get("success"))
            fail_count = len(results) - success_count

            total_entities = sum(
                item.get("entity_count", 0)
                for item in results
            )

            total_relationships = sum(
                item.get("relationship_count", 0)
                for item in results
            )

            print(
                "[Neo4j KG] LangGraph 构建完成："
                f"成功 {success_count} 个 chunk，"
                f"失败 {fail_count} 个 chunk，"
                f"实体 {total_entities} 个，"
                f"关系 {total_relationships} 条。"
            )

        except Exception as exc:
            print(f"[Neo4j KG] 构建失败：{exc}")