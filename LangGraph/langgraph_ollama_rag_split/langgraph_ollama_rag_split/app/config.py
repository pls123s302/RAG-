"""
项目配置集中放这里。
后面要换模型、换 Chroma 路径，只改这个文件。
"""

OLLAMA_URL = "http://localhost:11434"
GENERATE_MODEL = "qwen3:8b"             # 生成 / 路由 / 判断模型
EMBED_MODEL = "qwen3-embedding:0.6b"    # 嵌入模型

CHROMA_DB_PATH = "./chroma_db"          # Chroma 数据库路径
COLLECTION_NAME = "demo_rag"

RETRIEVE_TOP_K = 3
