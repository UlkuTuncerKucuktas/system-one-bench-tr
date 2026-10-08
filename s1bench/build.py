from . import DATA, save
from .tasks import TASKS


def build(*tasks):
    for name in tasks or TASKS:
        for split, items in TASKS[name]().items():
            save(DATA / name / f"{split}.jsonl", [{"id": f"{name}/{split}/{i:05d}", "task": name, "split": split, **item} for i, item in enumerate(items)])
            print(f"{name} {split}: {len(items)}")
