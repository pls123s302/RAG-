import hashlib
import json
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from neo4j import GraphDatabase

from app.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)


def make_chunk_id(doc: Document) -> str:
    """
    根据 chunk 文本和 metadata 生成稳定 ID。
    同一个 chunk 重复入库时，ID 基本保持一致，避免重复创建 Chunk 节点。
    """
    raw = doc.page_content + "\n" + json.dumps(
        doc.metadata,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def make_entity_id(name: str, entity_type: str) -> str:
    """
    实体唯一 ID。

    这里不用 Neo4j label 表示实体类型，而是统一用 :Entity，
    再用 type 属性区分人物、地点、武魂、魂兽、事件等。
    """
    raw = f"{entity_type}::{name}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class Neo4jKGStore:
    """
    Neo4j 知识图谱存储类。

    负责：
    1. 初始化 Neo4j 约束；
    2. 写入 Chunk 节点；
    3. 写入 Entity 节点；
    4. 写入 Entity 之间的关系；
    5. 写入 Chunk 与 Entity 的提及关系。
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        """
        初始化 Neo4j 约束。

        Entity 使用 id 唯一约束。
        Chunk 使用 id 唯一约束。
        """
        queries = [
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
            FOR (c:Chunk)
            REQUIRE c.id IS UNIQUE
            """,
        ]

        with self.driver.session(database=NEO4J_DATABASE) as session:
            for query in queries:
                session.run(query)

    def upsert_document_kg(self, doc: Document, kg_data: dict[str, Any]) -> None:
        """
        将一个 Document chunk 对应的知识图谱数据写入 Neo4j。

        kg_data 格式大致为：
        {
            "entities": [...],
            "relationships": [...]
        }
        """
        chunk_id = make_chunk_id(doc)
        metadata = doc.metadata or {}

        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                self._upsert_chunk_tx,
                chunk_id,
                doc.page_content,
                metadata,
            )

            for entity in kg_data.get("entities", []) or []:
                name = str(entity.get("name", "")).strip()
                entity_type = str(entity.get("type", "")).strip() or "其他"

                if not name:
                    continue

                entity_payload = {
                    "id": make_entity_id(name, entity_type),
                    "name": name,
                    "type": entity_type,
                    "description": str(entity.get("description", "")).strip(),
                }

                session.execute_write(
                    self._upsert_entity_tx,
                    entity_payload,
                )

                session.execute_write(
                    self._link_chunk_entity_tx,
                    chunk_id,
                    entity_payload["id"],
                )

            for rel in kg_data.get("relationships", []) or []:
                source = str(rel.get("source", "")).strip()
                target = str(rel.get("target", "")).strip()

                if not source or not target:
                    continue

                source_type = str(rel.get("source_type", "")).strip() or "其他"
                target_type = str(rel.get("target_type", "")).strip() or "其他"

                rel_payload = {
                    "source_id": make_entity_id(source, source_type),
                    "source": source,
                    "source_type": source_type,
                    "target_id": make_entity_id(target, target_type),
                    "target": target,
                    "target_type": target_type,
                    "type": str(rel.get("type", "")).strip() or "相关",
                    "description": str(rel.get("description", "")).strip(),
                    "evidence": str(rel.get("evidence", "")).strip(),
                    "chunk_id": chunk_id,
                }

                session.execute_write(
                    self._upsert_relationship_tx,
                    rel_payload,
                )

    @staticmethod
    def _upsert_chunk_tx(tx, chunk_id: str, text: str, metadata: dict) -> None:
        tx.run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.text = $text,
                c.metadata = $metadata,
                c.source = $source,
                c.chapter = $chapter
            """,
            id=chunk_id,
            text=text,
            metadata=json.dumps(metadata, ensure_ascii=False),
            source=str(metadata.get("source", "")),
            chapter=str(metadata.get("chapter", "")),
        )

    @staticmethod
    def _upsert_entity_tx(tx, entity: dict) -> None:
        tx.run(
            """
            MERGE (e:Entity {id: $id})
            SET e.name = $name,
                e.type = $type,
                e.description =
                    CASE
                        WHEN $description = "" THEN coalesce(e.description, "")
                        ELSE $description
                    END
            """,
            id=entity["id"],
            name=entity["name"],
            type=entity["type"],
            description=entity.get("description", ""),
        )

    @staticmethod
    def _link_chunk_entity_tx(tx, chunk_id: str, entity_id: str) -> None:
        tx.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {id: $entity_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_id=entity_id,
        )

    @staticmethod
    def _upsert_relationship_tx(tx, rel: dict) -> None:
        tx.run(
            """
            MERGE (s:Entity {id: $source_id})
            SET s.name = $source,
                s.type = $source_type

            MERGE (t:Entity {id: $target_id})
            SET t.name = $target,
                t.type = $target_type

            MERGE (s)-[r:RELATED_TO {type: $type}]->(t)
            SET r.description =
                    CASE
                        WHEN $description = "" THEN coalesce(r.description, "")
                        ELSE $description
                    END,
                r.evidence =
                    CASE
                        WHEN $evidence = "" THEN coalesce(r.evidence, "")
                        ELSE $evidence
                    END,
                r.sources =
                    CASE
                        WHEN r.sources IS NULL THEN [$chunk_id]
                        WHEN NOT $chunk_id IN r.sources THEN r.sources + $chunk_id
                        ELSE r.sources
                    END
            """,
            source_id=rel["source_id"],
            source=rel["source"],
            source_type=rel["source_type"],
            target_id=rel["target_id"],
            target=rel["target"],
            target_type=rel["target_type"],
            type=rel["type"],
            description=rel.get("description", ""),
            evidence=rel.get("evidence", ""),
            chunk_id=rel["chunk_id"],
        )

    def search_related_facts(
        self,
        entity_names: list[str],
        limit: int = 20,
    ) -> list[dict]:
        """
        根据实体名称，从 Neo4j 中召回相关关系。

        例如问题里抽到“唐三”，这里会查：
        唐三 --[关系]-- 其他实体
        """
        if not entity_names:
            return []

        with self.driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE any(name IN $names WHERE e.name CONTAINS name OR name CONTAINS e.name)

                OPTIONAL MATCH (e)-[r:RELATED_TO]-(n:Entity)

                RETURN
                    e.name AS entity_name,
                    e.type AS entity_type,
                    e.description AS entity_description,
                    r.type AS relation_type,
                    r.description AS relation_description,
                    r.evidence AS evidence,
                    n.name AS neighbor_name,
                    n.type AS neighbor_type
                LIMIT $limit
                """,
                names=entity_names,
                limit=limit,
            )

            return [dict(record) for record in result]


@lru_cache(maxsize=1)
def get_kg_store() -> Neo4jKGStore:
    """
    获取 Neo4j 知识图谱存储对象。

    lru_cache 可以避免每次调用都重复创建连接。
    """
    store = Neo4jKGStore()
    store.init_schema()
    return store