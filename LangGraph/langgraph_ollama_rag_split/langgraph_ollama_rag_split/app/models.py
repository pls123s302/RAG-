from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import (
    EMBED_MODEL,
    GENERATE_MODEL,
    OLLAMA_URL,
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    获取 Ollama 聊天模型。

    注意：
    这里不要写 think=False。
    有些版本的 ollama Python client 不支持 think 参数，
    会报：
    TypeError: Client.chat() got an unexpected keyword argument 'think'
    """
    return ChatOllama(
        model=GENERATE_MODEL,
        base_url=OLLAMA_URL,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """
    获取 Ollama 嵌入模型。
    """
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_URL,
    )