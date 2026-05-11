from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import add_documents


def build_demo_knowledge_base() -> None:
    """
    构建演示知识库。
    真实项目里，这个文件可以替换成 txt/pdf/docx 读取逻辑。
    """

    raw_docs = [
        Document(
            page_content="""
地穴魔蛛是《斗罗大陆》中的魂兽之一，具有较强的蛛网控制能力。
它擅长利用蛛网限制敌人的行动，并通过地形和伏击方式形成压制。
在相关剧情中，地穴魔蛛常被描述为危险、狡猾、难以正面对抗的魂兽。
""",
            metadata={"source": "斗罗大陆设定示例", "chapter": "魂兽资料"},
        ),
        Document(
            page_content="""
唐三是《斗罗大陆》的主要角色之一，拥有蓝银草和昊天锤双生武魂。
他的战斗方式强调控制、暗器、毒素以及战术判断。
""",
            metadata={"source": "斗罗大陆设定示例", "chapter": "人物资料"},
        ),
        Document(
            page_content="""
RAG 是检索增强生成技术。它通常先从知识库中检索相关内容，
再把检索到的上下文和用户问题一起交给大模型生成答案。
这样可以减少模型胡编，提高回答与知识库的一致性。
""",
            metadata={"source": "RAG 示例资料", "chapter": "技术说明"},
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(raw_docs)
    add_documents(split_docs)

    print(f"已写入 {len(split_docs)} 个切分块到 Chroma。")
