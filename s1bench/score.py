import csv
import json
import math
from collections import Counter

from .build import DATA

RESULTS = DATA.parent / "results"

TEMPERATURES = [0.05 * 1.1**i for i in range(80)]


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def pairs(model, task, split):
    items = {item["id"]: item for item in read(DATA / task / f"{split}.jsonl")}
    result = []
    for row in read(RESULTS / model / task / f"{split}.jsonl"):
        item = items[row["id"]]
        for qid, probs in row["probs"].items():
            if probs is not None:
                gold = item["gold"][qid]
                result.append((probs, str(gold).lower() if isinstance(gold, bool) else str(gold)))
    return result


def rescale(probs, temperature):
    weights = {key: max(p, 1e-12) ** (1 / temperature) for key, p in probs.items()}
    total = sum(weights.values())
    return {key: w / total for key, w in weights.items()}


def nll(data, temperature):
    return -sum(math.log(max(rescale(probs, temperature)[gold], 1e-12)) for probs, gold in data) / len(data)


def ece(data, bins=15):
    total = 0
    for b in range(bins):
        in_bin = [(max(p.values()), max(p, key=p.get) == g) for p, g in data if b / bins < max(p.values()) <= (b + 1) / bins]
        if in_bin:
            confidence = sum(c for c, _ in in_bin) / len(in_bin)
            accuracy = sum(ok for _, ok in in_bin) / len(in_bin)
            total += len(in_bin) / len(data) * abs(confidence - accuracy)
    return total


def macro_f1(data):
    predictions = [(max(p, key=p.get), gold) for p, gold in data]
    f1 = []
    for label in {gold for _, gold in data}:
        true_positives = sum(pred == label and gold == label for pred, gold in predictions)
        predicted = sum(pred == label for pred, _ in predictions)
        actual = sum(gold == label for _, gold in predictions)
        f1.append(2 * true_positives / (predicted + actual))
    return sum(f1) / len(f1)


def metrics(model, task):
    dev = pairs(model, task, "dev")
    temperature = min(TEMPERATURES, key=lambda t: nll(dev, t))
    test = [(rescale(probs, temperature), gold) for probs, gold in pairs(model, task, "test")]
    row = {
        "model": model,
        "task": task,
        "n": len(test),
        "accuracy": sum(max(p, key=p.get) == g for p, g in test) / len(test),
        "majority": max(Counter(g for _, g in test).values()) / len(test),
        "macro_f1": macro_f1(test),
        "brier": sum(sum((p[k] - (k == g)) ** 2 for k in p) for p, g in test) / len(test),
        "ece": ece(test),
        "temperature": temperature,
    }
    if all(key.isdigit() for key in test[0][0]):
        row["mae"] = sum(abs(sum(int(k) * v for k, v in p.items()) - int(g)) for p, g in test) / len(test)
    return row


def main():
    rows = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if (p / "test.jsonl").exists() and (p / "dev.jsonl").exists()):
            if pairs(model_dir.name, task_dir.name, "test"):
                rows.append(metrics(model_dir.name, task_dir.name))

    with open(RESULTS / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "n", "accuracy", "majority", "macro_f1", "brier", "ece", "mae", "temperature"])
        writer.writeheader()
        writer.writerows(rows)

    tasks = sorted({row["task"] for row in rows})
    models = sorted({row["model"] for row in rows})
    accuracy = {(row["model"], row["task"]): row["accuracy"] for row in rows}
    print("accuracy".ljust(22) + "".join(task[:10].rjust(11) for task in tasks))
    for model in models:
        cells = [f"{accuracy[(model, task)]:.3f}" if (model, task) in accuracy else "-" for task in tasks]
        print(model.ljust(22) + "".join(cell.rjust(11) for cell in cells))


if __name__ == "__main__":
    main()
