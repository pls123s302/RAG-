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

    如果 metadata 里已经有 chunk_id，则直接复用。
    这样可以保证 Chroma 和 Neo4j 使用同一个 chunk_id。
    """
    metadata = doc.metadata or {}
    existing_id = metadata.get("chunk_id")

    if existing_id:
        return str(existing_id)

    raw = doc.page_content + "\n" + json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def make_entity_id(name: str) -> str:
    """
    实体 ID 只由 name 生成，避免同名实体因为 type 不一致被拆成多个节点。
    例如：
    唐三 / 人物
    唐三 / Person
    唐三 / 其他

    都应该合并为同一个 Entity 节点。
    """
    raw = name.strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class Neo4jKGStore:
    """
    Neo4j 知识图谱存储。

    统一结构：
    - (:Chunk)
    - (:Entity)

    业务分类不作为 Neo4j label，而是放在 Entity.type 属性中。
    这样框架可以适配小说、论文、企业文档、医疗文档、法律文档等不同领域。
    """

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    def init_schema(self) -> None:
        queries = [
            """
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.name IS UNIQUE
            """,
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

        这里必须使用单个 execute_write。
        如果当前 chunk 的任意一个实体或关系写入失败，
        Neo4j 会回滚本 chunk 的全部写入。
        """
        chunk_id = make_chunk_id(doc)
        metadata = doc.metadata or {}

        entities = []
        for entity in kg_data.get("entities", []) or []:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "")).strip() or "其他"

            if not name:
                continue

            entities.append(
                {
                    "id": make_entity_id(name),
                    "name": name,
                    "type": entity_type,
                    "description": str(entity.get("description", "")).strip(),
                }
            )

        relationships = []
        for rel in kg_data.get("relationships", []) or []:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()

            if not source or not target:
                continue

            source_type = str(rel.get("source_type", "")).strip() or "其他"
            target_type = str(rel.get("target_type", "")).strip() or "其他"

            relationships.append(
                {
                    "source_id": make_entity_id(source),
                    "source": source,
                    "source_type": source_type,
                    "target_id": make_entity_id(target),
                    "target": target,
                    "target_type": target_type,
                    "type": str(rel.get("type", "")).strip() or "相关",
                    "description": str(rel.get("description", "")).strip(),
                    "evidence": str(rel.get("evidence", "")).strip(),
                    "chunk_id": chunk_id,
                }
            )

        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                self._upsert_document_kg_tx,
                chunk_id,
                doc.page_content,
                metadata,
                entities,
                relationships,
            )

    @staticmethod
    def _upsert_chunk_tx(tx, chunk_id: str, text: str, metadata: dict) -> None:
        tx.run(
            """
            MERGE (c:Chunk {id: $id})
            SET
                c.text = $text,
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
            MERGE (e:Entity {name: $name})
            SET
                e.id = coalesce(e.id, $id),
                e.type = CASE
                    WHEN coalesce(e.type, '') = '' OR e.type = '其他'
                    THEN $type
                    ELSE e.type
                END,
                e.description = CASE
                    WHEN $description = ''
                    THEN coalesce(e.description, '')
                    ELSE $description
                END
            """,
            id=entity["id"],
            name=entity["name"],
            type=entity["type"],
            description=entity.get("description", ""),
        )

    @staticmethod
    def _link_chunk_entity_tx(tx, chunk_id: str, entity_name: str) -> None:
        tx.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {name: $entity_name})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            chunk_id=chunk_id,
            entity_name=entity_name,
        )

    @staticmethod
    def _upsert_relationship_tx(tx, rel: dict) -> None:
        tx.run(
            """
            MERGE (s:Entity {name: $source})
            SET
                s.id = coalesce(s.id, $source_id),
                s.type = CASE
                    WHEN coalesce(s.type, '') = '' OR s.type = '其他'
                    THEN $source_type
                    ELSE s.type
                END

            MERGE (t:Entity {name: $target})
            SET
                t.id = coalesce(t.id, $target_id),
                t.type = CASE
                    WHEN coalesce(t.type, '') = '' OR t.type = '其他'
                    THEN $target_type
                    ELSE t.type
                END

            MERGE (s)-[r:RELATED_TO {type: $type}]->(t)
            SET
                r.description = CASE
                    WHEN $description = ''
                    THEN coalesce(r.description, '')
                    ELSE $description
                END,
                r.evidence = CASE
                    WHEN $evidence = ''
                    THEN coalesce(r.evidence, '')
                    ELSE $evidence
                END,
                r.sources = CASE
                    WHEN r.sources IS NULL
                    THEN [$chunk_id]
                    WHEN NOT $chunk_id IN r.sources
                    THEN r.sources + $chunk_id
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

    @staticmethod
    def _upsert_document_kg_tx(
            tx,
            chunk_id: str,
            text: str,
            metadata: dict,
            entities: list[dict],
            relationships: list[dict],
    ) -> None:
        """
        单 chunk 的 Neo4j 原子写入事务。
        """
        Neo4jKGStore._upsert_chunk_tx(
            tx=tx,
            chunk_id=chunk_id,
            text=text,
            metadata=metadata,
        )

        for entity in entities:
            Neo4jKGStore._upsert_entity_tx(
                tx=tx,
                entity=entity,
            )

            Neo4jKGStore._link_chunk_entity_tx(
                tx=tx,
                chunk_id=chunk_id,
                entity_name=entity["name"],
            )

        for rel in relationships:
            Neo4jKGStore._upsert_relationship_tx(
                tx=tx,
                rel=rel,
            )

    def rollback_document_kg(self, chunk_id: str) -> None:
        """
        回滚某个 chunk 已写入 Neo4j 的内容。

        用于这种场景：
        1. 当前 chunk 的 Neo4j 已写入成功；
        2. 但是 Chroma 写入失败；
        3. 为了保持两边一致，需要删除当前 chunk 在 Neo4j 中产生的痕迹。

        注意：
        - 会删除当前 Chunk；
        - 会删除 Chunk -> Entity 的 MENTIONS；
        - 会从 RELATED_TO.sources 中移除当前 chunk_id；
        - 如果某条关系只来自当前 chunk，则删除该关系；
        - 最后清理无引用、无关系的孤立 Entity。
        """
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.execute_write(
                self._rollback_document_kg_tx,
                chunk_id,
            )

    @staticmethod
    def _rollback_document_kg_tx(tx, chunk_id: str) -> None:
        tx.run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            DETACH DELETE c
            """,
            chunk_id=chunk_id,
        )

        tx.run(
            """
            MATCH ()-[r:RELATED_TO]-()
            WHERE $chunk_id IN coalesce(r.sources, [])
              AND size(coalesce(r.sources, [])) <= 1
            DELETE r
            """,
            chunk_id=chunk_id,
        )

        tx.run(
            """
            MATCH ()-[r:RELATED_TO]-()
            WHERE $chunk_id IN coalesce(r.sources, [])
            SET r.sources = [source IN r.sources WHERE source <> $chunk_id]
            """,
            chunk_id=chunk_id,
        )

        tx.run(
            """
            MATCH (e:Entity)
            WHERE NOT (e)<-[:MENTIONS]-(:Chunk)
              AND NOT (e)--()
            DELETE e
            """
        )


@lru_cache(maxsize=1)
def get_kg_store() -> Neo4jKGStore:
    store = Neo4jKGStore()
    store.init_schema()
    return store