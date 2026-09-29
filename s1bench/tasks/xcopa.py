from ..download import hf_rows
from ..split import split

INSTRUCTIONS = {
    "cause": "Bu durumun en olası SEBEBİ hangisidir?",
    "effect": "Bu durumun en olası SONUCU hangisidir?",
}


def to_items(split_name):
    items = []
    for r in hf_rows("cambridgeltl/xcopa", "tr", split_name):
        question = {"type": "choice", "instructions": INSTRUCTIONS[r["question"]], "criteria": {"A": r["choice1"], "B": r["choice2"]}}
        items.append({"state": r["premise"], "questions": {"answer": question}, "gold": {"answer": "AB"[r["label"]]}})
    return items


def build():
    return split(to_items("test"), to_items("validation"))
