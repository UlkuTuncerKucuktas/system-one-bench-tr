import json

from ..download import hf_rows
from ..split import split

CONFIGS = [
    "arc_challenge",
    "banking77",
    "boolq",
    "chaosnli",
    "civil_comments",
    "clinc150",
    "fever_evidence",
    "go_emotions",
    "helpsteer2_helpfulness",
    "helpsteer2_verbosity",
    "ledgar",
    "massive",
    "measuring_hate_speech",
    "mmlu",
    "mnli",
    "paws",
    "sms_spam",
    "sst5",
    "strategyqa_closed",
    "strategyqa_grounded",
    "stsb",
    "yelp5",
]


def votes(kind, soft):
    if kind == "noul":
        return {"true": soft, "false": 1 - soft}
    if kind == "score":
        return {str(level): p for level, p in enumerate(soft)}
    return soft


def to_item(r):
    question = json.loads(r["question"])
    if question["type"] == "choice":
        gold = r["label"]
    elif question["type"] == "noul":
        question.setdefault("criteria", {"true": "Evet", "false": "Hayır"})
        gold = r["label"] == "1"
    else:
        gold = int(r["label"])
    item = {"state": json.loads(r["state"]), "questions": {"answer": question}, "gold": {"answer": gold}}
    if r["soft_label"] is not None:
        item["soft_gold"] = {"answer": votes(question["type"], json.loads(r["soft_label"]))}
    return item


def build(config):
    return split([to_item(r) for r in hf_rows("hayriyigit/jev-bench-tr", config, "test")])
