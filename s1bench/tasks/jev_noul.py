import json

from ..download import hf_rows
from ..split import split


def to_items(split_name):
    items = []
    for r in hf_rows("ahmetege/turkish_jev_noul", None, split_name):
        items.append({
            "state": json.loads(r["state"]),
            "questions": json.loads(r["questions"]),
            "gold": {"q": json.loads(r["gold"])["q"]["label"] == "true"},
            "meta": {"workflow": r["workflow"], "inverted": r["inverted"]},
        })
    return items


def build():
    return split(to_items("test"), to_items("validation"))


def build_unseen():
    return split(to_items("test_unseen_task"))
