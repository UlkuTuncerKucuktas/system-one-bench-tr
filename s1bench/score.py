import csv
import json
import math
from collections import Counter

from .build import DATA
from .tasks import AREAS

RESULTS = DATA.parent / "results"

TEMPERATURES = [0.05 * 1.1**i for i in range(80)]
MIN_DEV = 30


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def answers(model, task, split):
    items = {item["id"]: item for item in read(DATA / task / f"{split}.jsonl")}
    by_question = {}
    for row in read(RESULTS / model / task / f"{split}.jsonl"):
        item = items[row["id"]]
        for qid, probs in row["probs"].items():
            if probs is not None:
                gold = item["gold"][qid]
                gold = str(gold).lower() if isinstance(gold, bool) else str(gold)
                by_question.setdefault(qid, []).append((probs, gold, item.get("soft_gold", {}).get(qid)))
    return by_question


def rescale(probs, temperature):
    weights = {key: max(p, 1e-12) ** (1 / temperature) for key, p in probs.items()}
    total = sum(weights.values())
    return {key: w / total for key, w in weights.items()}


def nll(data, temperature):
    return -sum(math.log(max(rescale(probs, temperature)[gold], 1e-12)) for probs, gold, _ in data) / len(data)


def ece(data, bins=15):
    total = 0
    for b in range(bins):
        in_bin = [(max(p.values()), max(p, key=p.get) == g) for p, g, _ in data if b / bins < max(p.values()) <= (b + 1) / bins]
        if in_bin:
            confidence = sum(c for c, _ in in_bin) / len(in_bin)
            accuracy = sum(ok for _, ok in in_bin) / len(in_bin)
            total += len(in_bin) / len(data) * abs(confidence - accuracy)
    return total


def brier(data):
    return sum(sum((p[k] - (k == g)) ** 2 for k in p) for p, g, _ in data) / len(data)


def macro_f1(data):
    predictions = [(max(p, key=p.get), gold) for p, gold, _ in data]
    f1 = []
    for label in {gold for _, gold in predictions}:
        true_positives = sum(pred == label and gold == label for pred, gold in predictions)
        predicted = sum(pred == label for pred, _ in predictions)
        actual = sum(gold == label for _, gold in predictions)
        f1.append(2 * true_positives / (predicted + actual))
    return sum(f1) / len(f1)


def metrics(dev, raw, score):
    temperature = min(TEMPERATURES, key=lambda t: nll(dev, t))
    test = [(rescale(probs, temperature), gold, soft) for probs, gold, soft in raw]
    row = {
        "n": len(test),
        "accuracy": sum(max(p, key=p.get) == g for p, g, _ in test) / len(test),
        "majority": max(Counter(g for _, g, _ in test).values()) / len(test),
        "macro_f1": macro_f1(test),
        "brier": brier(test),
        "ece": ece(test),
        "raw_nll": nll(raw, 1),
        "raw_brier": brier(raw),
        "raw_ece": ece(raw),
        "temperature": temperature,
    }
    if score:
        row["mae"] = sum(abs(sum(int(k) * v for k, v in p.items()) - int(g)) for p, g, _ in test) / len(test)
    if test[0][2]:
        row["vote_distance"] = sum(sum(abs(p[k] - soft[k]) for k in p) / 2 for p, _, soft in test) / len(test)
    return row


def main():
    rows = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if (p / "test.jsonl").exists() and (p / "dev.jsonl").exists()):
            dev = answers(model_dir.name, task_dir.name, "dev")
            pooled = [row for rows in dev.values() for row in rows]
            scores = {qid for item in read(DATA / task_dir.name / "test.jsonl") for qid, q in item["questions"].items() if q["type"] == "score"}
            for qid, test in answers(model_dir.name, task_dir.name, "test").items():
                # questions with few dev items (many in bev_tr) get the temperature of the whole task
                question_dev = dev.get(qid, [])
                question_dev = question_dev if len(question_dev) >= MIN_DEV else pooled
                rows.append({"model": model_dir.name, "task": task_dir.name, "question": qid, **metrics(question_dev, test, qid in scores)})

    with open(RESULTS / "scores.csv", "w", newline="") as f:
        fieldnames = ["model", "task", "question", "n", "accuracy", "majority", "macro_f1", "brier", "ece", "mae", "vote_distance", "raw_nll", "raw_brier", "raw_ece", "temperature"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # a task's accuracy weights its questions by their number of items
    by_task = {}
    for row in rows:
        correct, n = by_task.get((row["model"], row["task"]), (0, 0))
        by_task[(row["model"], row["task"])] = (correct + row["accuracy"] * row["n"], n + row["n"])
    print("accuracy".ljust(22) + "".join(area.rjust(15) for area in AREAS))
    for model in sorted({row["model"] for row in rows}):
        cells = []
        for tasks in AREAS.values():
            scores = [by_task[(model, task)][0] / by_task[(model, task)][1] for task in tasks if (model, task) in by_task]
            cells.append(f"{sum(scores) / len(scores):.3f}" if len(scores) == len(tasks) else "-")
        print(model.ljust(22) + "".join(cell.rjust(15) for cell in cells))


if __name__ == "__main__":
    main()
