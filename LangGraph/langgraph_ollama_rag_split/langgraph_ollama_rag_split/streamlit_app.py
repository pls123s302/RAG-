import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.graph import build_graph
from app.knowledge.build_demo_kb import build_demo_knowledge_base
from app.vectorstore import add_documents


st.set_page_config(
    page_title="RAG 知识库助手",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_rag_app():
    """
    缓存 LangGraph 应用，避免每次页面刷新都重新编译图。
    """
    return build_graph()


def init_session_state() -> None:
    """
    初始化 Streamlit 会话状态。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def decode_uploaded_file(uploaded_file) -> str:
    """
    读取上传的 txt / md 文件内容。

    优先 utf-8，失败后尝试 gbk。
    """
    raw = uploaded_file.read()

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gbk", errors="ignore")


def build_documents_from_uploads(uploaded_files) -> list[Document]:
    """
    将上传文件转换成 LangChain Document。
    """
    docs: list[Document] = []

    for uploaded_file in uploaded_files:
        text = decode_uploaded_file(uploaded_file)

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": uploaded_file.name,
                    "file_type": Path(uploaded_file.name).suffix,
                },
            )
        )

    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """
    切分上传文档。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    return splitter.split_documents(docs)


def format_metadata(metadata: dict[str, Any]) -> str:
    """
    格式化 metadata，便于页面展示。
    """
    if not metadata:
        return "无"

    return "\n".join(
        f"- {key}: {value}"
        for key, value in metadata.items()
    )


def render_docs(docs: list[Document]) -> None:
    """
    展示检索结果。
    """
    if not docs:
        st.info("本次没有检索到文档，可能走的是 DIRECT 直接回答。")
        return

    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}

        source = metadata.get("source", "unknown")

        with st.expander(f"资料 {index} | 来源：{source}", expanded=False):
            st.markdown("**Metadata：**")
            st.markdown(format_metadata(metadata))

            st.markdown("**Content：**")
            st.write(doc.page_content)


def invoke_rag(question: str) -> dict:
    """
    调用 LangGraph RAG 应用。
    """
    app = get_rag_app()

    return app.invoke(
        {
            "question": question,
            "route": "",
            "docs": [],
            "answer": "",
        }
    )


def render_sidebar() -> None:
    """
    左侧栏：知识库构建、文件上传、系统说明。
    """
    st.sidebar.title("📚 RAG 知识库助手")

    st.sidebar.markdown(
        """
        当前版本：
        - LangGraph 编排
        - Ollama 本地模型
        - Chroma 向量库
        - 可扩展 Neo4j 知识图谱
        - Streamlit 可视化界面
        """
    )

    st.sidebar.divider()

    st.sidebar.subheader("1. 构建演示知识库")

    if st.sidebar.button("构建 Demo 知识库", use_container_width=True):
        with st.spinner("正在构建演示知识库..."):
            try:
                build_demo_knowledge_base()
                st.sidebar.success("Demo 知识库构建完成。")
            except Exception as exc:
                st.sidebar.error(f"构建失败：{exc}")
                st.sidebar.code(traceback.format_exc())

    st.sidebar.divider()

    st.sidebar.subheader("2. 上传文本知识库")

    uploaded_files = st.sidebar.file_uploader(
        "上传 txt / md 文件",
        type=["txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.sidebar.button("写入上传文件", use_container_width=True):
            with st.spinner("正在切分并写入知识库..."):
                try:
                    docs = build_documents_from_uploads(uploaded_files)
                    split_docs = split_documents(docs)

                    add_documents(split_docs)

                    st.sidebar.success(
                        f"已写入 {len(split_docs)} 个 chunk。"
                    )
                except Exception as exc:
                    st.sidebar.error(f"写入失败：{exc}")
                    st.sidebar.code(traceback.format_exc())

    st.sidebar.divider()

    if st.sidebar.button("清空聊天记录", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


def render_chat_history() -> None:
    """
    渲染聊天记录。
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    init_session_state()
    render_sidebar()

    st.title("📖 RAG 可视化知识库助手")

    st.caption(
        "输入问题后，系统会通过 LangGraph 自动判断是直接回答，还是调用本地知识库检索。"
    )

    render_chat_history()

    question = st.chat_input("请输入你的问题，例如：唐三有哪些武魂？")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("正在思考并检索知识库..."):
                try:
                    result = invoke_rag(question)
                    answer = result.get("answer", "")
                    route = result.get("route", "")
                    docs = result.get("docs", [])

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                    st.session_state.last_result = result

                except Exception as exc:
                    error_text = f"运行出错：{exc}"
                    st.error(error_text)
                    st.code(traceback.format_exc())

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text,
                        }
                    )

                    return

    result = st.session_state.last_result

    if result:
        st.divider()

        left, right = st.columns([1, 3])

        with left:
            st.subheader("路由结果")
            st.code(result.get("route", ""))

            docs = result.get("docs", [])
            st.metric("检索资料数", len(docs))

        with right:
            st.subheader("检索资料")
            render_docs(result.get("docs", []))


if __name__ == "__main__":
    main()