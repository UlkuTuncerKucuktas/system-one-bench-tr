import json
import sys
from pathlib import Path

from .tasks import TASKS

DATA = Path(__file__).parent.parent / "data"


def build(name):
    for split, items in TASKS[name]().items():
        path = DATA / name / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, item in enumerate(items):
                item = {"id": f"{name}/{split}/{i:05d}", "task": name, "split": split, **item}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{name} {split}: {len(items)}")


if __name__ == "__main__":
    for name in sys.argv[1:] or TASKS:
        build(name)
