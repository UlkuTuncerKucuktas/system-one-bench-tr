from ..download import hf_rows
from ..split import split

TOPICS = ["siyaset", "dünya", "ekonomi", "kültür", "sağlık", "spor", "teknoloji"]
QUESTION = {"type": "choice", "instructions": "Bu haber hangi kategoriye giriyor?", "criteria": dict.fromkeys(TOPICS)}


def build():
    items = [
        {"state": r["text"].strip(), "questions": {"topic": QUESTION}, "gold": {"topic": TOPICS[r["category"]]}}
        for r in hf_rows("savasy/ttc4900", None, "train")
    ]
    return split(items)
