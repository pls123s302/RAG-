from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore import add_documents


def build_demo_knowledge_base() -> None:
    """
    构建演示知识库。

    当前 add_documents() 会执行两件事：
    1. 写入 Chroma 向量库；
    2. 调用 LangGraph KG 构建图，抽取实体关系并写入 Neo4j。
    """

    raw_docs = [
        Document(
            page_content="""
唐三是《斗罗大陆》的主要角色之一。他拥有蓝银草和昊天锤双生武魂。
蓝银草最初被认为是普通而弱小的武魂，但唐三后来逐渐展现出强大的控制能力。
昊天锤来自昊天宗，是力量极强的器武魂。
""".strip(),
            metadata={
                "source": "斗罗大陆演示知识库",
                "chapter": "人物设定-唐三",
            },
        ),
        Document(
            page_content="""
小舞是唐三的重要伙伴。她的本体是十万年魂兽柔骨兔。
小舞与唐三关系亲密，在剧情中多次与唐三并肩战斗。
她的身份与魂兽化形、献祭等重要剧情有关。
""".strip(),
            metadata={
                "source": "斗罗大陆演示知识库",
                "chapter": "人物设定-小舞",
            },
        ),
        Document(
            page_content="""
史莱克七怪是史莱克学院中的核心团队。
成员包括唐三、小舞、戴沐白、奥斯卡、马红俊、宁荣荣和朱竹清。
他们在学院学习和历练过程中形成了紧密的伙伴关系。
""".strip(),
            metadata={
                "source": "斗罗大陆演示知识库",
                "chapter": "组织设定-史莱克七怪",
            },
        ),
        Document(
            page_content="""
武魂殿是《斗罗大陆》中的重要组织。比比东是武魂殿教皇。
武魂殿与唐三、史莱克七怪之间存在多次冲突。
比比东和千仞雪都与武魂殿的权力体系密切相关。
""".strip(),
            metadata={
                "source": "斗罗大陆演示知识库",
                "chapter": "组织设定-武魂殿",
            },
        ),
        Document(
            page_content="""
地穴魔蛛是《斗罗大陆》中的魂兽之一，擅长使用蛛网限制敌人的行动。
它具有控制能力和伏击能力，经常依靠地形和蛛网压制对手。
在魂兽设定中，地穴魔蛛通常被描述为危险而狡猾的存在。
""".strip(),
            metadata={
                "source": "斗罗大陆演示知识库",
                "chapter": "魂兽设定-地穴魔蛛",
            },
        ),
        Document(
            page_content="""
RAG 是检索增强生成技术。它通常先从知识库中检索相关内容，
再把检索到的上下文和用户问题一起交给大模型生成答案。
这种方式可以减少大模型胡编，提高回答与知识库内容的一致性。
""".strip(),
            metadata={
                "source": "RAG 技术演示知识库",
                "chapter": "技术说明-RAG",
            },
        ),
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    split_docs = splitter.split_documents(raw_docs)

    add_documents(split_docs)

    print(f"已构建演示知识库，共写入 {len(split_docs)} 个 chunk。")
    print("如果 ENABLE_KNOWLEDGE_GRAPH=True，则这些 chunk 已同步抽取实体关系并写入 Neo4j。")