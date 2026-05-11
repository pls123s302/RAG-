from app.config import RETRIEVE_TOP_K
from app.state import RAGState
from app.vectorstore import get_vector_db


def retrieve_node(state: RAGState) -> RAGState:
    """
    检索节点：从 Chroma 中召回相关文档。
    """

    question = state["question"]
    vector_db = get_vector_db()

    docs = vector_db.similarity_search(
        question,
        k=RETRIEVE_TOP_K,
    )

    return {
        **state,
        "docs": docs,
        "route": "rag",
    }
