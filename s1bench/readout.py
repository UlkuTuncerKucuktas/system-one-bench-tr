from pathlib import Path

import torch

from . import DATA, RESULTS, read, save
from .llm import LLM
from .prompt import orders_of, render
from .tasks import MAIN

ITEMS = 100


@torch.no_grad()
def letter_log_probabilities(llm, texts, sizes):
    # over the whole vocabulary, before the letters are renormalized
    batch = llm.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    log_probs = llm.model(**batch, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
    return [[row[ids].logsumexp(0).item() for ids in llm.label_ids[:size]] for row, size in zip(log_probs, sizes)]


def readout(address):
    # the first test items of each main task, every question in each of its option orders
    llm = LLM(address)
    jobs = [
        (item, qid, options)
        for task in MAIN
        for item in read(DATA / task / "test.jsonl")[:ITEMS]
        for qid, question in item["questions"].items()
        for options in orders_of(question)
    ]
    texts = [llm.prompt(render(item, item["questions"][qid], options)) for item, qid, options in jobs]
    by_length = sorted(range(len(jobs)), key=lambda j: len(texts[j]))
    answers = [None] * len(jobs)
    for start in range(0, len(jobs), llm.batch_size):
        batch = by_length[start : start + llm.batch_size]
        results = letter_log_probabilities(llm, [texts[j] for j in batch], [len(jobs[j][2]) for j in batch])
        for j, log_probs in zip(batch, results):
            answers[j] = log_probs

    rows = {}
    for (item, qid, options), log_probs in zip(jobs, answers):
        orders = rows.setdefault(item["id"], {}).setdefault(qid, [])
        orders.append({key: x for (key, _), x in zip(options, log_probs)})
    save(RESULTS / Path(address).name / "readout.jsonl", [{"id": item_id, "log_probs": log_probs} for item_id, log_probs in rows.items()])
