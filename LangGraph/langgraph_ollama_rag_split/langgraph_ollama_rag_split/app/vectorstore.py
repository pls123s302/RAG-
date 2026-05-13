from functools import lru_cache
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    ENABLE_KNOWLEDGE_GRAPH,
    ENABLE_CHUNK_DEDUP,
    CHUNK_DEDUP_SIMILARITY_THRESHOLD,
    CHUNK_DEDUP_TOP_K,
)
from app.models import get_embeddings
from app.kg.neo4j_store import get_kg_store, make_chunk_id


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    """
    获取 Chroma 向量库对象。

    显式使用 cosine，方便用 0.95 这类相似度阈值做去重。
    如果你之前已经创建过旧的 chroma_db，建议删除后重建。
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _with_chunk_id(doc: Document, chunk_id: str) -> Document:
    """
    给 chunk 写入稳定 chunk_id，方便 Chroma 和 Neo4j 对齐。
    """
    metadata = dict(doc.metadata or {})
    metadata["chunk_id"] = chunk_id

    return Document(
        page_content=doc.page_content,
        metadata=metadata,
    )


def _exists_by_id(vector_db: Chroma, chunk_id: str) -> bool:
    """
    精确 ID 去重：同一个 chunk_id 已存在则跳过。
    """
    try:
        result = vector_db.get(ids=[chunk_id])
        return bool(result.get("ids"))
    except Exception:
        return False


def _max_similarity(vector_db: Chroma, doc: Document) -> float:
    """
    查询当前 chunk 与已有向量库内容的最大余弦相似度。

    Chroma 在 cosine space 下返回的是 cosine distance：
    distance 越小越相似，similarity = 1 - distance。
    """
    try:
        count = vector_db._collection.count()
        if count <= 0:
            return 0.0

        query_embedding = get_embeddings().embed_query(doc.page_content)
        n_results = min(CHUNK_DEDUP_TOP_K, count)

        result = vector_db._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances", "documents", "metadatas"],
        )

        distances = result.get("distances") or [[]]
        if not distances or not distances[0]:
            return 0.0

        best_distance = float(min(distances[0]))
        similarity = 1.0 - best_distance

        return max(0.0, min(1.0, similarity))
    except Exception as exc:
        print(f"[Chroma 去重] 相似度检查失败，默认不跳过：{exc}")
        return 0.0


def _is_duplicate_chunk(
    vector_db: Chroma,
    doc: Document,
    chunk_id: str,
) -> Tuple[bool, str]:
    """
    判断 chunk 是否重复。

    规则：
    1. chunk_id 已存在：跳过
    2. 与已有 chunk 最大相似度 >= 阈值：跳过
    """
    if _exists_by_id(vector_db, chunk_id):
        return True, f"chunk_id 已存在：{chunk_id}"

    if not ENABLE_CHUNK_DEDUP:
        return False, ""

    similarity = _max_similarity(vector_db, doc)

    if similarity >= CHUNK_DEDUP_SIMILARITY_THRESHOLD:
        return (
            True,
            f"相似度 {similarity:.4f} >= 阈值 {CHUNK_DEDUP_SIMILARITY_THRESHOLD}",
        )

    return False, f"相似度 {similarity:.4f}"


def _build_kg_for_one_doc(doc: Document) -> bool:
    """
    为单个 chunk 构建 Neo4j 知识图谱。

    注意：
    这里按单个 chunk 调用 LangGraph，保证每个 chunk 独立判断成功/失败。
    """
    from app.kg.graph import build_knowledge_graph_with_langgraph

    final_state = build_knowledge_graph_with_langgraph([doc])
    results = final_state.get("results", [])

    if not results:
        print("[Neo4j KG] 未返回处理结果。")
        return False

    success = all(item.get("success") for item in results)

    if not success:
        for item in results:
            if item.get("error"):
                print(f"[Neo4j KG] chunk 构建失败：{item.get('error')}")
        return False

    return True


def _rollback_kg(chunk_id: str) -> None:
    """
    Chroma 写入失败时，回滚已经写入 Neo4j 的当前 chunk。
    """
    try:
        store = get_kg_store()
        store.rollback_document_kg(chunk_id)
        print(f"[Neo4j KG] 已回滚 chunk：{chunk_id}")
    except Exception as exc:
        print(f"[Neo4j KG] 回滚失败，需要人工检查 chunk_id={chunk_id}：{exc}")


def _rollback_vector(vector_db: Chroma, chunk_id: str) -> None:
    """
    保险处理：如果 Chroma add_documents 部分成功后抛错，则删除当前 chunk。
    """
    try:
        vector_db.delete(ids=[chunk_id])
        print(f"[Chroma] 已回滚 chunk：{chunk_id}")
    except Exception:
        pass


def add_documents(docs: List[Document]) -> None:
    """
    添加文档到知识库。

    新流程：
    1. 对每个 chunk 生成稳定 chunk_id；
    2. 先查 Chroma，若相似度过高则跳过，不写 Chroma，也不写 Neo4j；
    3. 如果启用 KG，先写 Neo4j；
    4. Neo4j 成功后再写 Chroma；
    5. 如果 Chroma 写入失败，回滚当前 chunk 的 Neo4j 数据。

    这样保证单个 chunk 尽量满足：
    - Chroma 成功 + Neo4j 成功
    - 或者两边都不写
    """
    if not docs:
        print("没有需要写入的文档。")
        return

    vector_db = get_vector_db()

    added_count = 0
    skipped_count = 0
    failed_count = 0

    for index, raw_doc in enumerate(docs, start=1):
        chunk_id = make_chunk_id(raw_doc)
        doc = _with_chunk_id(raw_doc, chunk_id)

        print(f"\n[入库] 处理 chunk {index}/{len(docs)}，chunk_id={chunk_id}")

        is_duplicate, reason = _is_duplicate_chunk(
            vector_db=vector_db,
            doc=doc,
            chunk_id=chunk_id,
        )

        if is_duplicate:
            skipped_count += 1
            print(f"[入库] 跳过重复 chunk：{reason}")
            continue

        kg_written = False

        if ENABLE_KNOWLEDGE_GRAPH:
            kg_written = _build_kg_for_one_doc(doc)

            if not kg_written:
                failed_count += 1
                print("[入库] Neo4j 构建失败，本 chunk 不写入 Chroma。")
                continue

        try:
            vector_db.add_documents(
                documents=[doc],
                ids=[chunk_id],
            )
            added_count += 1
            print(f"[Chroma] 已写入 chunk：{chunk_id}")

        except Exception as exc:
            failed_count += 1
            print(f"[Chroma] 写入失败：{exc}")

            _rollback_vector(vector_db, chunk_id)

            if ENABLE_KNOWLEDGE_GRAPH and kg_written:
                _rollback_kg(chunk_id)

    print(
        "\n[入库] 完成："
        f"新增 {added_count} 个，"
        f"跳过重复 {skipped_count} 个，"
        f"失败 {failed_count} 个。"
    )