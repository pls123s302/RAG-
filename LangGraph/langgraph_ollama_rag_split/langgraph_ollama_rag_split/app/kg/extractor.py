import json
from typing import Any

from app.kg.json_utils import extract_json_object
from app.models import get_llm


def build_kg_extraction_prompt(text: str, metadata: dict | None = None) -> str:
    """
    构建知识图谱抽取 Prompt。

    这里不固定实体类型，因为你后面可能会处理：
    人物、地点、组织、武魂、魂兽、魂技、物品、事件、概念、等级、称号等很多类型。
    """

    metadata_text = json.dumps(
        metadata or {},
        ensure_ascii=False,
    )

    return f"""
你是一个知识图谱抽取器，负责从文本中抽取实体和实体之间的关系。

请从下面文本中抽取：
1. 实体 entities
2. 实体之间的关系 relationships

要求：
1. 实体类型可以自由扩展，不要局限于固定类型。
2. 如果是小说文本，常见实体类型包括：
   人物、组织、地点、武魂、魂兽、魂技、物品、事件、概念、等级、称号、其他。
3. 关系类型要简短，例如：
   拥有、属于、师从、敌对、位于、使用、击败、相关、身份是、能力是、出现于。
4. 只抽取文本中明确支持的信息。
5. 不要编造文本中没有的信息。
6. 必须只输出合法 JSON。
7. 不要输出 Markdown。
8. 不要输出解释说明。

输出格式必须严格如下：

{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "实体类型",
      "description": "基于文本的简短描述"
    }}
  ],
  "relationships": [
    {{
      "source": "源实体名称",
      "source_type": "源实体类型",
      "target": "目标实体名称",
      "target_type": "目标实体类型",
      "type": "关系类型",
      "description": "关系说明",
      "evidence": "支持该关系的原文短句"
    }}
  ]
}}

文档元数据：
{metadata_text}

待抽取文本：
{text}
"""


def _clean_text(value: Any) -> str:
    """
    清洗模型输出里的字段。
    """
    return str(value or "").strip()


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    根据 name + type 去重实体。
    """
    seen = set()
    result = []

    for entity in entities:
        name = _clean_text(entity.get("name"))
        entity_type = _clean_text(entity.get("type")) or "其他"

        if not name:
            continue

        key = (name, entity_type)

        if key in seen:
            continue

        seen.add(key)

        result.append(
            {
                "name": name,
                "type": entity_type,
                "description": _clean_text(entity.get("description")),
            }
        )

    return result


def _deduplicate_relationships(relationships: list[dict]) -> list[dict]:
    """
    根据 source + target + type 去重关系。
    """
    seen = set()
    result = []

    for rel in relationships:
        source = _clean_text(rel.get("source"))
        target = _clean_text(rel.get("target"))
        rel_type = _clean_text(rel.get("type")) or "相关"

        if not source or not target:
            continue

        source_type = _clean_text(rel.get("source_type")) or "其他"
        target_type = _clean_text(rel.get("target_type")) or "其他"

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
    """
    从文本中抽取知识图谱数据。

    返回格式：
    {
        "entities": [...],
        "relationships": [...]
    }
    """

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