import time
from pathlib import Path

from . import DATA, RESULTS, read, save
from .jev import Jev
from .llm import LLM
from .tasks import TASKS


def run(address, *tasks):
    model = Jev() if address == "jev" else LLM(address)
    name = Path(address).name
    for task in tasks or TASKS:
        for split in ["dev", "test"]:
            items = read(DATA / task / f"{split}.jsonl")
            start = time.time()
            predictions = model.predict(items)
            save(RESULTS / name / task / f"{split}.jsonl", [{"id": item["id"], "probs": probs} for item, probs in zip(items, predictions)])
            print(f"{name} {task} {split}: {len(items)} items in {time.time() - start:.0f}s", flush=True)
