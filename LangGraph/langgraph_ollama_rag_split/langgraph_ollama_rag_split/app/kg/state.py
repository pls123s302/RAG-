from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document


class KGBuildState(TypedDict):
    """
    知识图谱构建流程的 LangGraph 状态。

    docs:
        待处理的文档 chunk 列表。

    current_index:
        当前处理到第几个 chunk。

    current_doc:
        当前正在处理的 Document。

    kg_data:
        当前 chunk 抽取出来的实体和关系。

    results:
        每个 chunk 的处理结果统计。

    error:
        当前流程中的错误信息。
    """

    docs: list[Document]
    current_index: int
    current_doc: Document | None
    kg_data: dict[str, Any]
    results: list[dict[str, Any]]
    error: str | None
