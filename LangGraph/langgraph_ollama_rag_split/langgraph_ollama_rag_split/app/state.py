from typing import TypedDict, List

from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    LangGraph 中每个节点之间传递的状态。
    """

    question: str
    route: str
    docs: List[Document]
    answer: str
