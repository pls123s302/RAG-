from app.kg.extractor import extract_kg_from_text
from app.kg.neo4j_store import get_kg_store
from app.kg.state import KGBuildState


def prepare_next_doc_node(state: KGBuildState) -> KGBuildState:
    """
    准备当前要处理的 Document。

    根据 current_index 从 docs 中取出当前 chunk。
    """

    docs = state.get("docs", [])
    current_index = state.get("current_index", 0)

    if current_index >= len(docs):
        return {
            **state,
            "current_doc": None,
            "error": None,
        }

    current_doc = docs[current_index]

    return {
        **state,
        "current_doc": current_doc,
        "kg_data": {
            "entities": [],
            "relationships": [],
        },
        "error": None,
    }


def extract_kg_node(state: KGBuildState) -> KGBuildState:
    """
    从当前 Document chunk 中抽取实体和关系。

    这里会调用本地 Ollama 模型。
    """

    current_doc = state.get("current_doc")

    if current_doc is None:
        return {
            **state,
            "error": "current_doc is None，无法抽取知识图谱。",
        }

    try:
        kg_data = extract_kg_from_text(
            text=current_doc.page_content,
            metadata=current_doc.metadata,
        )

        return {
            **state,
            "kg_data": kg_data,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "kg_data": {
                "entities": [],
                "relationships": [],
            },
            "error": f"抽取知识图谱失败：{exc}",
        }


def write_kg_node(state: KGBuildState) -> KGBuildState:
    """
    将当前 chunk 的实体和关系写入 Neo4j。
    """

    current_doc = state.get("current_doc")
    kg_data = state.get("kg_data", {})

    if current_doc is None:
        return {
            **state,
            "error": "current_doc is None，无法写入 Neo4j。",
        }

    # 如果上一步抽取失败，这里直接跳过写入
    if state.get("error"):
        return state

    try:
        store = get_kg_store()
        store.upsert_document_kg(
            doc=current_doc,
            kg_data=kg_data,
        )

        return {
            **state,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"写入 Neo4j 失败：{exc}",
        }


def record_result_node(state: KGBuildState) -> KGBuildState:
    """
    记录当前 chunk 的处理结果，并将 current_index + 1。
    """

    current_index = state.get("current_index", 0)
    kg_data = state.get("kg_data", {})
    results = state.get("results", [])

    entities = kg_data.get("entities", []) or []
    relationships = kg_data.get("relationships", []) or []

    result = {
        "chunk_index": current_index,
        "entity_count": len(entities),
        "relationship_count": len(relationships),
        "success": state.get("error") is None,
        "error": state.get("error"),
    }

    print(
        f"[Neo4j KG] chunk {current_index + 1}: "
        f"entities={len(entities)}, "
        f"relationships={len(relationships)}, "
        f"success={result['success']}"
    )

    if result["error"]:
        print(f"[Neo4j KG] error: {result['error']}")

    return {
        **state,
        "results": results + [result],
        "current_index": current_index + 1,
        "current_doc": None,
        "kg_data": {
            "entities": [],
            "relationships": [],
        },
        "error": None,
    }


def should_continue(state: KGBuildState) -> str:
    """
    LangGraph 条件边函数。

    如果还有 chunk 没处理完，继续；
    否则结束。
    """

    docs = state.get("docs", [])
    current_index = state.get("current_index", 0)

    if current_index < len(docs):
        return "continue"

    return "end"