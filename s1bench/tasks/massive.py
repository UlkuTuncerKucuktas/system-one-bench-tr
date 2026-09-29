from ..download import hf_rows
from ..split import split

INSTRUCTIONS = "Kullanıcı bu komutla sesli asistandan ne istiyor? Komutun niyetini (intent) seç."


def build():
    rows = {name: hf_rows("mteb/amazon_massive_intent", "tr", name) for name in ["train", "validation", "test"]}
    intents = dict.fromkeys(sorted({r["label"] for r in rows["train"]}))
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": intents}

    def to_items(split_rows):
        return [{"state": r["text"], "questions": {"intent": question}, "gold": {"intent": r["label"]}} for r in split_rows]

    return split(to_items(rows["test"]), to_items(rows["validation"]), to_items(rows["train"]))
