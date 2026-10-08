import re

from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "choice",
    "instructions": "Bu tweet nefret söylemi mi, saldırgan mı, yoksa normal mi?",
    "criteria": {
        "nefret": "Birini ya da bir grubu kökeni, dini, cinsiyeti, cinsel yönelimi ya da siyasi görüşü gibi bir kimliği yüzünden aşağılıyor ya da ona saldırıyor.",
        "saldırgan": "Küfür, hakaret ya da kaba ifade içeriyor ama bir kimliği hedef almıyor.",
        "normal": "Nefret ya da saldırganlık içermiyor.",
    },
}
LABELS = ["normal", "saldırgan", "nefret"]


def build():
    samples = {0: [], 1: []}
    for r in hf_rows("manueltonneau/hateday", None, "train"):
        if r["lang_country_hateday"] == "tr":
            state = re.sub(r"@\w+", "@USER", r["text"])
            samples[r["weighted"]].append({"state": state, "questions": {"hate": QUESTION}, "gold": {"hate": LABELS[r["class_clean"]]}})
    return split(samples[0], rest=samples[1], test_size=10000)
