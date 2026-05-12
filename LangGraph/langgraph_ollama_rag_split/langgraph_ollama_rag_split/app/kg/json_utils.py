import json
import re
from typing import Any


def strip_think_tags(text: str) -> str:
    """
    去掉模型可能输出的 <think>...</think> 内容。

    一些推理模型会输出思考过程，例如：
    <think>这里是模型思考</think>
    {"entities": [...]}

    我们只保留真正需要解析的部分。
    """
    return re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL,
    ).strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """
    从模型输出中提取 JSON 对象。

    支持两种情况：

    1. 模型直接输出纯 JSON：
       {"entities": [], "relationships": []}

    2. 模型输出了额外解释：
       下面是抽取结果：
       {"entities": [], "relationships": []}

    如果解析失败，返回空字典。
    """
    text = strip_think_tags(text)

    # 第一种情况：模型输出本身就是合法 JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 第二种情况：从文本中截取第一个 {...} JSON 对象
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}

    return {}