from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import CHROMA_DB_PATH, COLLECTION_NAME
from app.models import get_embeddings


@lru_cache(maxsize=1)
def get_vector_db() -> Chroma:
    """
    获取 Chroma 向量库对象。
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DB_PATH,
    )


def add_documents(docs: List[Document]) -> None:
    """
    向 Chroma 写入文档。
    """

    vector_db = get_vector_db()
    vector_db.add_documents(docs)
