from ..download import hf_rows
from ..split import split


def build():
    items = []
    for r in hf_rows("facebook/belebele", "tur_Latn", "test"):
        options = dict(zip("ABCD", [r["mc_answer1"], r["mc_answer2"], r["mc_answer3"], r["mc_answer4"]]))
        items.append({
            "state": r["flores_passage"],
            "questions": {"answer": {"type": "choice", "instructions": r["question"], "criteria": options}},
            "gold": {"answer": "ABCD"[int(r["correct_answer_num"]) - 1]},
        })
    return split(items)
