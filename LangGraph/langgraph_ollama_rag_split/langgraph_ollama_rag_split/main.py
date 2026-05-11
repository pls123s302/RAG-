import argparse

from app.graph import build_graph
from app.knowledge.build_demo_kb import build_demo_knowledge_base


def print_result(result: dict) -> None:
    print("\n========== 路由结果 ==========")
    print(result["route"])

    print("\n========== 检索结果 ==========")
    if result["docs"]:
        for i, doc in enumerate(result["docs"], start=1):
            print(f"\n--- 文档 {i} ---")
            print("metadata:", doc.metadata)
            print("content:", doc.page_content[:300])
    else:
        print("未调用 RAG。")

    print("\n========== 最终回答 ==========")
    print(result["answer"])


def run_cli() -> None:
    app = build_graph()

    print("LangGraph + Ollama + Chroma RAG 示例已启动。输入 exit 退出。")

    while True:
        question = input("\n用户问题：").strip()

        if question.lower() in ["exit", "quit", "q"]:
            break

        result = app.invoke(
            {
                "question": question,
                "route": "",
                "docs": [],
                "answer": "",
            }
        )

        print_result(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-demo",
        action="store_true",
        help="写入演示知识库。第一次运行可以加这个参数，后面不需要重复加。",
    )

    args = parser.parse_args()

    if args.build_demo:
        build_demo_knowledge_base()

    run_cli()


if __name__ == "__main__":
    main()
