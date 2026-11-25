import time
from pathlib import Path

from . import DATA, RESULTS, read, save
from .api import jev, openrouter
from .llm import LLM, VLLM, DeepSeek
from .local import Turnike, d3, karar, laya_tr, pusula, tau
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
    if address == "berkay123001/turnike-jev-4b":
        return Turnike()
    if address == "vllm-sr/d3":
        return d3()
    if address == "erenextlabs/pusula":
        return pusula()
    if address == "TurkishCodeMan/laya-tr":
        return laya_tr()
    if address == "vllm/deepseek-ai/DeepSeek-V4-Flash":
        return DeepSeek()
    if address.startswith("vllm/"):
        return VLLM(address.removeprefix("vllm/"))
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
