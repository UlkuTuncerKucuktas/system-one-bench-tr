import csv
import math
from collections import Counter

from . import DATA, RESULTS, read
from .tasks import AREAS, MAIN, SECONDARY

TEMPERATURES = [0.05 * 1.1**i for i in range(80)]
MIN_DEV = 30
# jev-bench-tr tasks not every model answered: more options than the 62 letters (banking77, clinc150, ledgar) or than
# Metask's 26 (go_emotions, massive)
TOO_MANY_OPTIONS = ["jevbench_banking77_tr", "jevbench_clinc150_tr", "jevbench_ledgar_tr", "jevbench_go_emotions_tr", "jevbench_massive_tr"]
SUMMARY = {
    "accuracy": (MAIN, "accuracy"),
    "macro-F1": (MAIN, "macro_f1"),
    "Brier": (MAIN, "brier"),
    "ECE": (MAIN, "ece"),
    "raw ECE": (MAIN, "raw_ece"),
    "raw log-loss": (MAIN, "raw_nll"),
    "Score MAE": (["buyuksinema_tr", "musteri_yorumlari_tr", "stsb_tr"], "mae"),
    "MASSIVE": (["massive_intent_tr"], "accuracy"),
    "secondary": ([task for task in SECONDARY if not task.startswith("jevbench_")], "accuracy"),
    "jev-bench-tr": ([task for task in SECONDARY if task.startswith("jevbench_") and task not in TOO_MANY_OPTIONS], "accuracy"),
}


def answers(model, task, split):
    items = {item["id"]: item for item in read(DATA / task / f"{split}.jsonl")}
    by_question = {}
    for row in read(RESULTS / model / task / f"{split}.jsonl"):
        item = items[row["id"]]
        for qid, probs in row["probs"].items():
            # Metask's stored results hold null for questions with more than its 26 options
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
    for label in sorted({gold for _, gold in predictions}):
        true_positives = sum(pred == label and gold == label for pred, gold in predictions)
        predicted = sum(pred == label for pred, _ in predictions)
        actual = sum(gold == label for _, gold in predictions)
        f1.append(2 * true_positives / (predicted + actual))
    return sum(f1) / len(f1)


def metrics(dev, raw, is_score):
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
    if is_score:
        row["mae"] = sum(abs(sum(int(k) * v for k, v in p.items()) - int(g)) for p, g, _ in test) / len(test)
    if test[0][2]:
        row["vote_distance"] = sum(sum(abs(p[k] - soft[k]) for k in p) / 2 for p, _, soft in test) / len(test)
    return row


def task_value(questions, column):
    return sum(q[column] * q["n"] for q in questions) / sum(q["n"] for q in questions)


def table(rows, groups):
    by_task = {}
    for row in rows:
        by_task.setdefault((row["model"], row["task"]), []).append(row)
    print(" " * 22 + "".join(name.rjust(15) for name in groups))
    for model in sorted({row["model"] for row in rows}):
        cells = []
        for tasks, column in groups.values():
            values = [task_value(by_task[model, task], column) for task in tasks if (model, task) in by_task]
            cells.append(f"{sum(values) / len(values):.3f}" if len(values) == len(tasks) else "-")
        print(model.ljust(22) + "".join(cell.rjust(15) for cell in cells))


def score():
    rows = []
    for model_dir in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in model_dir.iterdir() if (p / "test.jsonl").exists() and (p / "dev.jsonl").exists()):
            dev = answers(model_dir.name, task_dir.name, "dev")
            task_dev = [example for examples in dev.values() for example in examples]
            score_qids = {qid for item in read(DATA / task_dir.name / "test.jsonl") for qid, q in item["questions"].items() if q["type"] == "score"}
            for qid, test in answers(model_dir.name, task_dir.name, "test").items():
                # questions with few dev items (many in bev_tr) get the temperature of the whole task
                question_dev = dev.get(qid, [])
                question_dev = question_dev if len(question_dev) >= MIN_DEV else task_dev
                rows.append({"model": model_dir.name, "task": task_dir.name, "question": qid, **metrics(question_dev, test, qid in score_qids)})

    with open(RESULTS / "scores.csv", "w", newline="") as f:
        fieldnames = ["model", "task", "question", "n", "accuracy", "majority", "macro_f1", "brier", "ece", "mae", "vote_distance", "raw_nll", "raw_brier", "raw_ece", "temperature"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table(rows, SUMMARY)
    print("\naccuracy by area")
    table(rows, {area: (tasks, "accuracy") for area, tasks in AREAS.items()})
