from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import OLLAMA_URL, GENERATE_MODEL, EMBED_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    获取生成模型。
    使用缓存，避免每个节点重复初始化。
    """

    return ChatOllama(
        base_url=OLLAMA_URL,
        model=GENERATE_MODEL,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """
    获取嵌入模型。
    """

    return OllamaEmbeddings(
        base_url=OLLAMA_URL,
        model=EMBED_MODEL,
    )
