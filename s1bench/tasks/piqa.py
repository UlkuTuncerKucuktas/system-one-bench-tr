from ..download import hf_rows
from ..split import split

INSTRUCTIONS = "Bu ifadeyi hangisi daha mantıklı biçimde tamamlıyor?"


def build():
    items = []
    for r in hf_rows("mrlbenchmarks/global-piqa-nonparallel", "tur_latn", "test"):
        question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {"A": r["solution0"], "B": r["solution1"]}}
        items.append({"state": r["prompt"], "questions": {"answer": question}, "gold": {"answer": "AB"[r["label"]]}})
    return split(items)
