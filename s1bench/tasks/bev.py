import json

from ..download import hf_rows
from ..split import split


def to_item(r):
    questions = json.loads(r["questions_json"])
    for question in questions.values():
        if question["type"] == "noul":
            question.setdefault("criteria", {"true": "Evet", "false": "Hayır"})
    return {
        "state": r["state"],
        "questions": {qid: {key: value for key, value in q.items() if key != "label"} for qid, q in questions.items()},
        "gold": {qid: q["label"] for qid, q in questions.items()},
        "meta": {"domain": r["domain"]},
    }


def build():
    return split([to_item(r) for r in hf_rows("hayriyigit/bev-decision-150K-tr", None, "test")])
