import time
from pathlib import Path

from . import DATA, RESULTS, read, save
from .api import jev, openrouter
from .llm import LLM
from .local import karar, tau
from .tasks import TASKS


def model_for(address):
    if address == "jev":
        return jev()
    if address.startswith("openrouter/"):
        return openrouter(address.removeprefix("openrouter/"))
    if address == "ufakai/ufakzeka-karar":
        return karar()
    if address == "cisimcik/tau-4b":
        return tau()
    return LLM(address)


def run(address, *tasks):
    model = model_for(address)
    name = Path(address).name
    for task in tasks or TASKS:
        for split in ["dev", "test"]:
            items = read(DATA / task / f"{split}.jsonl")
            start = time.time()
            predictions = model.predict(items)
            save(RESULTS / name / task / f"{split}.jsonl", [{"id": item["id"], "probs": probs} for item, probs in zip(items, predictions)])
            print(f"{name} {task} {split}: {len(items)} items in {time.time() - start:.0f}s", flush=True)
