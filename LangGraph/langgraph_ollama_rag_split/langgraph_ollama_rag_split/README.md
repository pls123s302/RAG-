# LangGraph + Ollama + Chroma RAG 拆分版

## 目录结构

```text
langgraph_ollama_rag_split/
├── main.py
├── requirements.txt
└── app/
    ├── config.py
    ├── state.py
    ├── models.py
    ├── vectorstore.py
    ├── prompts.py
    ├── graph.py
    ├── knowledge/
    │   └── build_demo_kb.py
    └── nodes/
        ├── router.py
        ├── retrieve.py
        └── answer.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 拉取 Ollama 模型

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

## 第一次运行：写入演示知识库

```bash
python main.py --build-demo
```

## 后续运行

```bash
python main.py
```

## 当前流程

```text
START
  └── route_question
        ├── rag    -> retrieve -> rag_answer -> END
        └── direct -> direct_answer -> END
```
