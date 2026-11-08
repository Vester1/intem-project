import argparse
import json
from pathlib import Path
from .graph import Application
from .models import Question


def main():
    parser = argparse.ArgumentParser(description="Intem · прототип КП2")
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("ask", help="Один вопрос без памяти диалога")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Ответ, источники и маршрут агентов")
    commands.add_parser("graph", help="Базовый граф в формате Mermaid")
    commands.add_parser("index-core", help="Заново индексировать четыре локальных PDF")
    args = parser.parse_args()
    app = Application()
    if args.command == "ask":
        result = app.ask(Question(question=args.question))
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else result["answer"])
    elif args.command == "graph":
        print(app.graph.get_graph().draw_mermaid())
    else:
        manifest = json.loads(Path("data/mephi/manifest.json").read_text(encoding="utf-8"))
        for item in manifest:
            count = app.retrieval.ingest(item["file"], source=item["url"], metadata=item)
            print(f'{count} фрагментов: {item["title"]}')


if __name__ == "__main__":
    main()
