import csv
import json
import zlib
from collections import Counter

import numpy as np

from . import DATA, RESULTS, read
from .tasks import AREAS, MAIN

GROUPS = {
    "leading_hosted_decision": ["jev", "microsoft-decision-1"],
    "open_decision": ["d3", "pusula", "tau-4b", "turnike-jev-4b", "metask-jev-4b", "ufakzeka-karar", "laya-multilingual", "laya-tr"],
    "general_llm": ["gemma-4-31b", "gemma-4-12b", "gemma-4-e4b", "qwen3.5-397b-a17b", "qwen3.8-27b", "qwen3.5-35b-a3b", "qwen3.5-9b",
                    "qwen3.5-4b", "deepseek-v4-flash", "eurollm-9b"],
    "turkish_llm": ["trendyol-asure-12b", "turkish-gemma-9b", "kizagan-e4b", "mecellem-qwen3-4b", "kumru-2b"],
}
SYSTEMS = [model for models in GROUPS.values() for model in models]
GROUP_OF = {model: group for group, models in GROUPS.items() for model in models}
# the decimals of the two systems that return rounded probabilities
ROUNDED = {"jev": 2, "laya-multilingual": 4}
PAIRS = [
    ("d3", "jev"), ("d3", "microsoft-decision-1"), ("pusula", "d3"), ("microsoft-decision-1", "jev"), ("tau-4b", "jev"),
    ("d3", "qwen3.8-27b"), ("microsoft-decision-1", "qwen3.5-9b"), ("tau-4b", "qwen3.5-4b"), ("turnike-jev-4b", "qwen3.5-4b"),
    ("metask-jev-4b", "qwen3.5-4b"), ("trendyol-asure-12b", "gemma-4-12b"), ("turkish-gemma-9b", "qwen3.5-9b"),
    ("kizagan-e4b", "gemma-4-e4b"), ("kumru-2b", "qwen3.5-4b"), ("tau-4b", "gemma-4-12b"), ("gemma-4-31b", "jev"),
]
TURKISH_TEXT = [
    "news_topic_tr", "buyuksinema_tr", "musteri_yorumlari_tr", "biltweetnews_tr", "tweet_emotion_tr", "offenseval_tr", "sms_spam_tr",
    "hateday_tr", "xfact_teyit_tr", "checkthat_tr", "trclaim19_tr", "webfaq_tr", "touche_tr", "turkishmmlu_tr", "turblimp_tr",
]
NOT_BY_PEOPLE = ["tweet_emotion_tr", "webfaq_tr", "touche_tr", "sms_spam_tr", "turblimp_tr"]
SUBSETS = {
    "main": MAIN,
    "turkish_text": TURKISH_TEXT,
    "turkish_text_labels_by_people": [task for task in TURKISH_TEXT if task not in NOT_BY_PEOPLE],
}
MAIN_AREAS = {area: [task for task in tasks if task in MAIN] for area, tasks in AREAS.items()}
MEASURES = ["accuracy", "brier", "log_loss", "log_loss_t"]
DRAWS = 2000
FLOOR = 1e-12
BOUNDS = (0.05, 20.0)
SEARCH_STEP = 1e-6
SURE = 0.9
WRONG = 0.01
TARGETS = [0.9, 0.95]
COVERAGES = [round(0.1 * k, 2) for k in range(1, 11)]
EDGES = np.arange(1, 10) / 10


def average(values):
    return sum(values) / len(values)


def gold_of(item, qid):
    gold = item["gold"][qid]
    return str(gold).lower() if isinstance(gold, bool) else str(gold)


class Block:
    def __init__(self, model, rows):
        self.items, self.qids, self.kinds, self.keys, self.gold_keys, answers, soft = zip(*rows)
        shape = (len(rows), max(map(len, answers)))
        self.probs, self.filled, self.levels = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.soft = np.zeros(shape) if all(soft) else None
        for i, row in enumerate(rows):
            k, probs = len(row[3]), np.array(row[5], dtype=float)
            total = probs.sum()
            if model in ROUNDED:
                # without float noise a returned 0.90 is 0.9 and equal values are tied
                probs, total = probs.round(ROUNDED[model]), total.round(ROUNDED[model])
            self.probs[i, :k] = probs / total
            if model in ROUNDED:
                # a returned zero counts as half the rounding step
                probs[probs == 0] = 0.5 * 10 ** -ROUNDED[model]
            self.filled[i, :k] = probs / probs.sum()
            if self.soft is not None:
                self.soft[i, :k] = [row[6][key] for key in row[3]]
            if row[2] == "score":
                self.levels[i, :k] = [int(key) for key in row[3]]
        self.gold = np.array([row[3].index(row[4]) for row in rows])
        self.predicted = self.probs.argmax(1)
        self.right = (self.predicted == self.gold).astype(float)
        self.top = self.probs.max(1)
        with np.errstate(divide="ignore"):
            self.log_probs = np.log(self.filled)

    def log_loss(self, probs):
        return -np.log(np.clip(probs[np.arange(len(self.gold)), self.gold], FLOOR, None))

    def brier(self):
        target = np.zeros_like(self.probs)
        target[np.arange(len(self.gold)), self.gold] = 1
        return ((self.probs - target) ** 2).sum(1)


def load(model, task, split):
    items = {item["id"]: item for item in read(DATA / task / f"{split}.jsonl")}
    rows = []
    for row in read(RESULTS / model / task / f"{split}.jsonl"):
        item = items[row["id"]]
        for qid, probs in row["probs"].items():
            # Metask's stored results hold null for questions with more than its 26 options
            if probs is not None:
                rows.append((row["id"], qid, item["questions"][qid]["type"], list(probs), gold_of(item, qid), list(probs.values()), item.get("soft_gold", {}).get(qid)))
    return Block(model, rows) if rows else None


def tempered(log_probs, temperature):
    z = log_probs / temperature
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_temperature(blocks):
    count = sum(len(block.gold) for block in blocks)

    def loss(log_t):
        return sum(block.log_loss(tempered(block.log_probs, np.exp(log_t))).sum() for block in blocks) / count

    # golden-section search on log T
    ratio = (5**0.5 - 1) / 2
    low, high = np.log(BOUNDS[0]), np.log(BOUNDS[1])
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    loss_left, loss_right = loss(left), loss(right)
    while high - low > SEARCH_STEP:
        if loss_left < loss_right:
            high, right, loss_right = right, left, loss_left
            left = high - ratio * (high - low)
            loss_left = loss(left)
        else:
            low, left, loss_left = left, right, loss_right
            right = low + ratio * (high - low)
            loss_right = loss(right)
    temperature = float(np.exp((low + high) / 2))
    # a temperature at a bound is the search's limit, not a fit
    return None if temperature <= BOUNDS[0] * 1.01 or temperature >= BOUNDS[1] * 0.99 else temperature


def summary(value, draws):
    return {"value": float(value), "interval": np.percentile(draws, [2.5, 97.5]).tolist()}


def bootstrap(task):
    ids = sorted(item["id"] for item in read(DATA / task / "test.jsonl"))
    # the same draws for every system
    picks = np.random.default_rng(zlib.crc32(task.encode())).integers(0, len(ids), size=(DRAWS, len(ids)))
    return {item_id: n for n, item_id in enumerate(ids)}, picks


def resample(draws, item_ids, values):
    index, picks = draws
    item = np.array([index[item_id] for item_id in item_ids])
    count = np.bincount(item, minlength=len(index))
    sums = np.bincount(item, weights=values, minlength=len(index))
    return values.mean(), sums[picks].sum(1) / count[picks].sum(1)


def outcomes_of(question):
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "noul":
        return ["true", "false"]
    return [str(level) for level in range(len(question["criteria"]))]


# every test question gets the label frequencies of its question on dev, with one added to each
def reference(task, draws):
    counts = {}
    for item in read(DATA / task / "dev.jsonl"):
        for qid, question in item["questions"].items():
            count = counts.setdefault(qid, dict.fromkeys(outcomes_of(question), 1))
            count[gold_of(item, qid)] = count.get(gold_of(item, qid), 1) + 1
    ids, loss, brier, right = [], [], [], []
    for item in read(DATA / task / "test.jsonl"):
        for qid, question in item["questions"].items():
            count = counts.get(qid) or dict.fromkeys(outcomes_of(question), 1)
            gold, total = gold_of(item, qid), sum(count.values())
            ids.append(item["id"])
            loss.append(-np.log(count.get(gold, 1) / total))
            brier.append(sum((n / total - (key == gold)) ** 2 for key, n in count.items()))
            right.append(float(max(count, key=count.get) == gold))
    values = {"accuracy": right, "brier": brier, "log_loss": loss}
    return {measure: resample(draws, ids, np.array(x)) for measure, x in values.items()}


def majorities():
    rates = {}
    for task in MAIN:
        golds = {}
        for item in read(DATA / task / "test.jsonl"):
            for qid in item["questions"]:
                golds.setdefault(qid, []).append(gold_of(item, qid))
        n = sum(map(len, golds.values()))
        rates[task] = sum(Counter(g).most_common(1)[0][1] for g in golds.values()) / n, n
    rate = lambda tasks: average([rates[task][0] for task in tasks])
    return {
        "tasks": {task: {"rate": rate, "test_questions": n} for task, (rate, n) in rates.items()},
        "areas": {area: rate(tasks) for area, tasks in MAIN_AREAS.items()},
        "main": rate(MAIN),
        "subsets": {name: rate(tasks) for name, tasks in SUBSETS.items()},
    }


def bins(confidence, right):
    where = np.digitize(confidence, EDGES)
    return np.array([[(where == k).sum(), confidence[where == k].sum(), right[where == k].sum()] for k in range(10)]) / len(right)


def reliability(total):
    share, confidence, hits = (total / len(MAIN)).T
    # an empty bin has no mean and no accuracy
    mean_top = [float(c / w) if w > 0 else None for c, w in zip(confidence, share)]
    accuracy = [float(h / w) if w > 0 else None for h, w in zip(hits, share)]
    return {"share": share.tolist(), "mean_top_probability": mean_top, "accuracy": accuracy, "ece": float(np.abs(hits - confidence).sum())}


# the lowest top probability of the dev questions at which the dev questions with at least that probability reach the target
def threshold(dev, target):
    reached = [v for v in np.unique(dev.top) if dev.right[dev.top >= v].mean() >= target]
    return min(reached) if reached else np.inf


def decided(test, dev, target):
    chosen = test.top >= threshold(dev, target)
    return chosen.mean(), test.right[chosen].mean() if chosen.any() else None


# accuracy of the k most confident answers; answers tied at the cut count with the share of them that fits
def coverage_curve(test):
    out = []
    for c in COVERAGES:
        k = max(1, round(c * len(test.gold)))
        cut = np.sort(test.top)[::-1][k - 1]
        above, tied = test.top > cut, test.top == cut
        out.append((test.right[above].sum() + test.right[tied].sum() * (k - above.sum()) / tied.sum()) / k)
    return np.array(out)


def deciding_alone(rows, target):
    reached = [accuracy for _, accuracy in rows if accuracy is not None]
    return {
        "coverage": float(average([share for share, _ in rows])),
        "accuracy": float(average(reached)) if reached else None,
        "tasks_reaching_target": int(sum(accuracy >= target for accuracy in reached)),
        "tasks_deciding_nothing": len(rows) - len(reached),
    }


def evaluate(tests, devs, draws):
    temperature = fit_temperature([devs[task] for task in MAIN])
    point, drawn = {}, {}
    total = {"raw": np.zeros((10, 3)), "t": np.zeros((10, 3))}
    alone = {target: [] for target in TARGETS}
    curve, sure, wrong = np.zeros(len(COVERAGES)), np.zeros(2), np.zeros(3)
    for task in MAIN:
        test = tests[task]
        after = tempered(test.log_probs, temperature) if temperature else test.filled
        loss = test.log_loss(test.filled)
        values = {"accuracy": test.right, "brier": test.brier(), "log_loss": loss, "log_loss_t": test.log_loss(after)}
        for measure, x in values.items():
            point[task, measure], drawn[task, measure] = resample(draws[task], test.items, x)
        for name, probs in [("raw", test.probs), ("t", after)]:
            total[name] += bins(probs.max(1), test.right)
        high = test.top >= SURE
        sure += [high.mean(), test.right[high].sum() / len(test.gold)]
        for target in TARGETS:
            alone[target].append(decided(test, devs[task], target))
        curve += coverage_curve(test)
        bad = test.probs[np.arange(len(test.gold)), test.gold] < WRONG
        wrong += [bad.mean(), loss[bad].sum() / len(test.gold), loss.mean()]
    row = {
        "temperature": temperature,
        "reliability": {name: reliability(x) for name, x in total.items()},
        "top_at_least_0.9": {"share": float(sure[0] / len(MAIN)), "accuracy": float(sure[1] / sure[0]) if sure[0] else None},
        "sure_wrong": {"share_of_questions": float(wrong[0] / len(MAIN)), "share_of_log_loss": float(wrong[1] / wrong[2])},
        "alone": {str(target): deciding_alone(rows, target) for target, rows in alone.items()},
        "coverage_curve": dict(zip(map(str, COVERAGES), (curve / len(MAIN)).tolist())),
    }
    return row, point, drawn


def describe(point, drawn, measures):
    over = lambda tasks, m: summary(average([point[task, m] for task in tasks]), average([drawn[task, m] for task in tasks]))
    return {
        "subsets": {name: {m: over(tasks, m) for m in measures} for name, tasks in SUBSETS.items()},
        "areas": {area: {m: over(tasks, m) for m in measures} for area, tasks in MAIN_AREAS.items()},
        "tasks": {task: {m: over([task], m) for m in measures} for task in MAIN},
    }


# a rank is one plus the number of systems that are better, so ties share the best rank
def ranks(point, drawn):
    out = {model: {} for model in SYSTEMS}
    for measure in MEASURES:
        sign = -1 if measure == "accuracy" else 1
        value = sign * np.array([average([point[model][task, measure] for task in MAIN]) for model in SYSTEMS])
        draw = sign * np.array([average([drawn[model][task, measure] for task in MAIN]) for model in SYSTEMS])
        point_rank, draw_rank = 1 + (value[None] < value[:, None]).sum(1), 1 + (draw[None] < draw[:, None]).sum(1)
        for i, model in enumerate(SYSTEMS):
            out[model][measure] = {"rank": int(point_rank[i]), "interval": np.percentile(draw_rank[i], [2.5, 97.5]).tolist()}
    return out


def pair_differences(point, drawn):
    return {
        name: {
            f"{a} - {b}": {
                m: summary(average([point[a][t, m] - point[b][t, m] for t in tasks]), average([drawn[a][t, m] - drawn[b][t, m] for t in tasks]))
                for m in MEASURES
            }
            for a, b in PAIRS
        }
        for name, tasks in SUBSETS.items()
    }


def question_rows(model, task, test):
    brier, loss = test.brier(), test.log_loss(test.filled)
    expected = (test.probs * test.levels).sum(1)
    by_question = {}
    for i, qid in enumerate(test.qids):
        by_question.setdefault(qid, []).append(i)
    rows = []
    for qid, at in by_question.items():
        gold = [test.gold_keys[i] for i in at]
        predicted = [test.keys[i][test.predicted[i]] for i in at]
        f1 = [2 * sum(p == label and g == label for p, g in zip(predicted, gold)) / (predicted.count(label) + gold.count(label)) for label in sorted(set(gold))]
        row = {
            "model": model, "task": task, "question": qid, "n": len(at), "accuracy": test.right[at].mean(),
            "majority": max(Counter(gold).values()) / len(gold), "macro_f1": sum(f1) / len(f1), "brier": brier[at].mean(), "log_loss": loss[at].mean(),
        }
        if test.kinds[at[0]] == "score":
            row["mae"] = np.abs(expected[at] - [int(g) for g in gold]).mean()
        if test.soft is not None:
            row["vote_distance"] = (np.abs(test.probs[at] - test.soft[at]).sum(1) / 2).mean()
        rows.append(row)
    return rows


def tables_of(systems, pairs, draws):
    tables = {name: {} for name in ["main", "confidence", "deciding_alone", "areas", "tasks", "subsets"]}
    for model, r in systems.items():
        main = r["subsets"]["main"]
        tables["main"][model] = {
            "group": GROUP_OF.get(model), "accuracy": main["accuracy"], "brier": main["brier"], "log_loss": main["log_loss"],
            "temperature": r["temperature"], "log_loss_t": main["log_loss_t"], "rank": r.get("rank"),
        }
        tables["confidence"][model] = {
            "top_at_least_0.9": r["top_at_least_0.9"], "sure_wrong": r["sure_wrong"], "reliability": r["reliability"],
        }
        tables["deciding_alone"][model] = {"accuracy_at_coverage": r["coverage_curve"], "dev_thresholds": r["alone"]}
        tables["areas"][model], tables["tasks"][model] = r["areas"], r["tasks"]
        tables["subsets"][model] = {name: m for name, m in r["subsets"].items() if name != "main"}
    point, drawn = {}, {}
    for task in MAIN:
        for measure, (value, draw) in reference(task, draws[task]).items():
            point[task, measure], drawn[task, measure] = value, draw
    row = describe(point, drawn, [m for m in MEASURES if m != "log_loss_t"])
    # the reference row has no temperature
    subsets = {name: {**m, "log_loss_t": None} for name, m in row["subsets"].items()}
    tables["subsets"]["reference"] = {name: m for name, m in subsets.items() if name != "main"}
    return {**tables, "pairs": pairs, "majority": majorities(), "reference": {"subsets": subsets, "areas": row["areas"], "tasks": row["tasks"]}}


def answered(model_dir):
    tasks = sorted(path.name for path in model_dir.iterdir() if (path / "test.jsonl").exists())
    return tasks if all((model_dir / task / split).exists() for task in MAIN for split in ["dev.jsonl", "test.jsonl"]) else None


def table(rows):
    print("| Model | Accuracy | Raw log-loss | " + " | ".join(area.replace("_", "-").capitalize() for area in MAIN_AREAS) + " |")
    print("|---|---|---|" + "---|" * len(MAIN_AREAS))
    for model, values in sorted(rows.items(), key=lambda row: -row[1][0]):
        print(f"| {model} | {100 * values[0]:.1f} | {values[1]:.2f} | " + " | ".join(f"{100 * v:.1f}" for v in values[2:]) + " |")


def score():
    draws = {task: bootstrap(task) for task in MAIN}
    rows, point, drawn, systems, readme = [], {}, {}, {}, {}
    for model_dir in sorted(path for path in RESULTS.iterdir() if path.is_dir()):
        tasks = answered(model_dir)
        if tasks is None:
            continue
        model = model_dir.name
        tests = {task: load(model, task, "test") for task in tasks}
        tests = {task: test for task, test in tests.items() if test}
        for task, test in tests.items():
            rows += question_rows(model, task, test)
        systems[model], point[model], drawn[model] = evaluate(tests, {task: load(model, task, "dev") for task in MAIN}, draws)
        systems[model].update(describe(point[model], drawn[model], MEASURES))
        readme[model] = [average([point[model][task, m] for task in MAIN]) for m in ["accuracy", "log_loss"]]
        readme[model] += [average([point[model][task, "accuracy"] for task in tasks]) for tasks in MAIN_AREAS.values()]
    for model, rank in ranks(point, drawn).items():
        systems[model]["rank"] = rank
    tables = tables_of(systems, pair_differences(point, drawn), draws)
    (RESULTS / "tables.json").write_text(json.dumps(tables, indent=1))

    with open(RESULTS / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "question", "n", "accuracy", "majority", "macro_f1", "brier", "log_loss", "mae", "vote_distance"])
        writer.writeheader()
        writer.writerows(rows)
    table(readme)
