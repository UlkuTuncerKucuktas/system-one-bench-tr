from ..download import hf_rows
from ..split import split

WORKFLOWS = ["enerji_dagitim", "kargo_teslimat", "klinik_yonlendirme", "ogrenci_isleri", "yazilim_destek"]
SPEAKERS = {"user": "Müşteri", "system": "Asistan"}


def build():
    items = []
    for workflow in WORKFLOWS:
        for r in hf_rows("moganai/lavoir-dialogues-tr", workflow, "test_zeroshot"):
            items.append({
                "state": "\n".join(f"{SPEAKERS[turn['role']]}: {turn['text']}" for turn in r["state"]),
                "questions": {"route": r["question"]},
                "gold": {"route": r["gold"]},
                "soft_gold": {"route": r["target"]},
                "meta": {"workflow": workflow, "kind": r["kind"]},
            })
    return split(items)
