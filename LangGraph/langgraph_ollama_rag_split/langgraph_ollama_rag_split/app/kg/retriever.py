import json

from langchain_core.documents import Document

from app.config import GRAPH_RETRIEVE_TOP_K
from app.kg.json_utils import extract_json_object
from app.kg.neo4j_store import get_kg_store
from app.models import get_llm


def build_query_entity_prompt(question: str) -> str:
    """
    构建问题实体抽取 Prompt。

    作用：
    从用户问题中抽取适合查询 Neo4j 的核心实体名称。

    例如：
    问题：唐三有哪些武魂？
    输出：{"entities": ["唐三"]}

    问题：小舞和唐三是什么关系？
    输出：{"entities": ["小舞", "唐三"]}
    """

    return f"""
你是一个查询实体抽取器。

请从用户问题中抽取适合查询知识图谱的核心实体名称。

要求：
1. 只抽取明确出现或明显指向的实体。
2. 实体可以是人物、地点、组织、武魂、魂兽、魂技、物品、事件、概念等。
3. 不要抽取泛泛的词，比如“关系”“能力”“原因”“介绍”“有哪些”。
4. 必须只输出合法 JSON。
5. 不要输出 Markdown。
6. 不要输出解释说明。

输出格式：

{{
  "entities": ["实体1", "实体2"]
}}

用户问题：
{question}
"""


def extract_query_entities(question: str) -> list[str]:
    """
    从用户问题中抽取查询实体。

    这里仍然使用本地 Ollama 模型。
    """

    prompt = build_query_entity_prompt(question)

    response = get_llm().invoke(prompt)
    raw_output = response.content

    data = extract_json_object(raw_output)

    entities = data.get("entities", []) or []

    result = []

    for entity in entities:
        name = str(entity).strip()

        if not name:
            continue

        if name not in result:
            result.append(name)

    return result


def _format_graph_fact(record: dict) -> str:
    """
    把 Neo4j 查询结果格式化成自然语言上下文。
    """

    entity_name = record.get("entity_name") or ""
    entity_type = record.get("entity_type") or "实体"

    neighbor_name = record.get("neighbor_name") or ""
    neighbor_type = record.get("neighbor_type") or "实体"

    relation_type = record.get("relation_type") or ""
    relation_description = record.get("relation_description") or ""
    entity_description = record.get("entity_description") or ""
    evidence = record.get("evidence") or ""

    # 没有邻居实体时，只返回实体说明
    if not neighbor_name:
        text = f"{entity_name}（{entity_type}）"
        if entity_description:
            text += f"：{entity_description}"
        return text

    # 有关系时，返回三元组形式
    text = (
        f"{entity_name}（{entity_type}）"
        f" --[{relation_type}]-- "
        f"{neighbor_name}（{neighbor_type}）"
    )

    if relation_description:
        text += f"；关系说明：{relation_description}"

    if evidence:
        text += f"；证据：{evidence}"

    return text


def retrieve_graph_context(question: str) -> list[Document]:
    """
    从 Neo4j 知识图谱中召回和问题相关的实体关系。

    返回值是 Document 列表，方便后面和 Chroma 的检索结果合并。
    """

    entity_names = extract_query_entities(question)

    if not entity_names:
        return []

    store = get_kg_store()

    records = store.search_related_facts(
        entity_names=entity_names,
        limit=GRAPH_RETRIEVE_TOP_K,
    )

    if not records:
        return []

    facts = []

    for record in records:
        fact = _format_graph_fact(record)
        if fact and fact not in facts:
            facts.append(fact)

    if not facts:
        return []

    page_content = "\n".join(
        f"{index + 1}. {fact}"
        for index, fact in enumerate(facts)
    )

    metadata = {
        "source": "neo4j_knowledge_graph",
        "query_entities": json.dumps(entity_names, ensure_ascii=False),
    }

    return [
        Document(
            page_content=page_content,
            metadata=metadata,
        )
    ]