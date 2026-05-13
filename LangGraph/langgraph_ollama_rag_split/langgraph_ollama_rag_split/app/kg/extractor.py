import json
from typing import Any

from app.kg.json_utils import extract_json_object
from app.models import get_llm


TYPE_ALIASES = {
    "person": "人物",
    "people": "人物",
    "character": "人物",
    "人物": "人物",
    "角色": "人物",

    "location": "地点",
    "locayion": "地点",
    "place": "地点",
    "地点": "地点",
    "地名": "地点",

    "organization": "组织",
    "organisation": "组织",
    "org": "组织",
    "组织": "组织",
    "机构": "组织",

    "concept": "概念",
    "概念": "概念",

    "event": "事件",
    "事件": "事件",

    "item": "物品",
    "object": "物品",
    "物品": "物品",
}


def normalize_entity_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "其他"

    key = text.lower()
    return TYPE_ALIASES.get(key, TYPE_ALIASES.get(text, text))


def build_kg_extraction_prompt(text: str, metadata: dict | None = None) -> str:
    metadata_text = json.dumps(metadata or {}, ensure_ascii=False)

    return f"""
你是通用知识图谱抽取器。请从文本中抽取实体和实体关系。

规则：
1. 只抽取文本明确支持的信息，不要推测。
2. 实体 name 使用原文中的稳定名称，避免同义改写。
3. type 使用简短中文词；可用通用类型如：人物、地点、组织、事件、物品、概念、时间；也可使用领域类型。
4. 不要输出 Neo4j label，不要输出 Person、Location、Organization、Concept 作为类型。
5. 关系 type 使用简短动词或关系词，如：属于、位于、拥有、使用、导致、包含、相关。
6. evidence 填写支持关系的原文短句。
7. 只输出合法 JSON，不要 Markdown，不要解释。

JSON 格式：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "description": "简短描述"}}
  ],
  "relationships": [
    {{
      "source": "源实体",
      "source_type": "源实体类型",
      "target": "目标实体",
      "target_type": "目标实体类型",
      "type": "关系类型",
      "description": "关系说明",
      "evidence": "原文证据"
    }}
  ]
}}

文档元数据：
{metadata_text}

文本：
{text}
"""


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    按 name 去重，不再按 name + type 去重。
    """
    seen = set()
    result = []

    for entity in entities:
        name = _clean_text(entity.get("name"))
        entity_type = normalize_entity_type(entity.get("type"))

        if not name:
            continue

        if name in seen:
            continue

        seen.add(name)
        result.append(
            {
                "name": name,
                "type": entity_type,
                "description": _clean_text(entity.get("description")),
            }
        )

    return result


def _deduplicate_relationships(relationships: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for rel in relationships:
        source = _clean_text(rel.get("source"))
        target = _clean_text(rel.get("target"))
        rel_type = _clean_text(rel.get("type")) or "相关"

        if not source or not target:
            continue

        source_type = normalize_entity_type(rel.get("source_type"))
        target_type = normalize_entity_type(rel.get("target_type"))

        key = (source, target, rel_type)
        if key in seen:
            continue

        seen.add(key)
        result.append(
            {
                "source": source,
                "source_type": source_type,
                "target": target,
                "target_type": target_type,
                "type": rel_type,
                "description": _clean_text(rel.get("description")),
                "evidence": _clean_text(rel.get("evidence")),
            }
        )

    return result


def extract_kg_from_text(text: str, metadata: dict | None = None) -> dict:
    prompt = build_kg_extraction_prompt(
        text=text,
        metadata=metadata,
    )

    response = get_llm().invoke(prompt)
    raw_output = response.content
    data = extract_json_object(raw_output)

    entities = data.get("entities", []) or []
    relationships = data.get("relationships", []) or []

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": _deduplicate_relationships(relationships),
    }