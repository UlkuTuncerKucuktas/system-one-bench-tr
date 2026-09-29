import json
import sys
import time
from pathlib import Path

from .build import DATA
from .models import MODELS

RESULTS = Path(__file__).parent.parent / "results"


def evaluate(model_name, tasks):
    model = MODELS[model_name]()
    for task in tasks:
        for split in ["dev", "test"]:
            items = [json.loads(line) for line in open(DATA / task / f"{split}.jsonl", encoding="utf-8")]
            start = time.time()
            predictions = model.predict(items)
            path = RESULTS / model_name / task / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for item, probs in zip(items, predictions):
                    f.write(json.dumps({"id": item["id"], "probs": probs}, ensure_ascii=False) + "\n")
            print(f"{model_name} {task} {split}: {len(items)} items in {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    model_name, *tasks = sys.argv[1:]
    evaluate(model_name, tasks or sorted(path.name for path in DATA.iterdir()))
