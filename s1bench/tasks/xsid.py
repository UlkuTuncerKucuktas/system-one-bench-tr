from ..download import download
from ..split import split

BASE = "https://raw.githubusercontent.com/mainlp/xsid/main/data/xSID-0.6/"
INSTRUCTIONS = "Kullanıcı bu komutla sesli asistandan ne istiyor? Komutun niyetini (intent) seç."


def examples(filename):
    pairs = []
    for block in download(BASE + filename).read_text(encoding="utf-8").strip().split("\n\n"):
        fields = dict(line[2:].split(" = ", 1) for line in block.splitlines() if line.startswith("# "))
        pairs.append((fields["text"], fields["intent"]))
    return pairs


def build():
    test, dev, train = examples("tr.test.conll"), examples("tr.valid.conll"), examples("tr.projectedTrain.conll")
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict.fromkeys(sorted({intent for _, intent in test + dev}))}

    def to_items(pairs):
        return [{"state": text, "questions": {"intent": question}, "gold": {"intent": intent}} for text, intent in pairs]

    return split(to_items(test), to_items(dev), to_items(train))
