"""
项目配置集中放这里。
后面要换模型、换 Chroma 路径，只改这个文件。
"""

OLLAMA_URL = "http://192.168.1.104:11434"
GENERATE_MODEL = "qwen3:8b"             # 生成 / 路由 / 判断模型
EMBED_MODEL = "qwen3-embedding:0.6b"    # 嵌入模型

CHROMA_DB_PATH = "./chroma_db"          # Chroma 数据库路径
COLLECTION_NAME = "demo_rag"

RETRIEVE_TOP_K = 10

# ========== Neo4j 知识图谱配置 ==========

# 本地 Neo4j 默认 Bolt 地址一般是这个
NEO4J_URI = "bolt://localhost:7687"

# Neo4j 用户名，默认一般是 neo4j
NEO4J_USERNAME = "neo4j"

# 改成你本地 Neo4j 的真实密码
NEO4J_PASSWORD = "password"

# 默认数据库名一般是 neo4j
NEO4J_DATABASE = "neo4j"


# ========== 知识图谱开关 ==========

# True：添加知识库时，同步抽取实体关系并写入 Neo4j
# False：只写 Chroma，不写 Neo4j
ENABLE_KNOWLEDGE_GRAPH = True

# 查询时，从 Neo4j 召回多少条实体关系
GRAPH_RETRIEVE_TOP_K = 20

# ========== 入库去重配置 ==========
ENABLE_CHUNK_DEDUP = True

# 当新 chunk 和已有 chunk 的余弦相似度 >= 该阈值时，跳过入库
CHUNK_DEDUP_SIMILARITY_THRESHOLD = 0.95

# 查最相似的几个已有 chunk
CHUNK_DEDUP_TOP_K = 3