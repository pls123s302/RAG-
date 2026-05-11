def build_route_prompt(question: str) -> str:
    return f"""
你是一个路由判断器。

你的任务是判断用户问题是否需要查询本地知识库。

如果问题涉及：
1. 小说、文档、知识库中的具体内容；
2. 专有名词；
3. 需要根据已有资料回答；
4. 可能需要检索上下文；

则回答：RAG

如果问题是普通闲聊、简单常识、代码解释、无需知识库的问题，则回答：DIRECT

只允许输出 RAG 或 DIRECT，不要输出其他内容。

用户问题：
{question}
"""


def build_rag_answer_prompt(question: str, context: str) -> str:
    return f"""
你是一个基于知识库回答问题的助手。

请严格根据下面的检索资料回答用户问题。
如果资料中没有答案，请明确说明“知识库中没有找到足够信息”，不要胡编。

检索资料：
{context}

用户问题：
{question}

请用中文回答。
"""


def build_direct_answer_prompt(question: str) -> str:
    return f"""
请直接回答用户问题，不需要调用知识库。

用户问题：
{question}

请用中文回答。
"""
