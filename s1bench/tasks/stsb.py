from ..download import download, read_tsv
from ..split import split

BASE = "https://raw.githubusercontent.com/verimsu/STSb-TR/main/data_splits/"
QUESTION = {
    "type": "score",
    "instructions": "İki cümle anlamca ne kadar benzer?",
    "criteria": [
        "0: İki cümle tamamen farklı konular hakkında.",
        "1: Anlamları farklı ama aynı konudan bahsediyorlar.",
        "2: Anlamları farklı ama bazı ayrıntıları paylaşıyorlar.",
        "3: Kabaca aynı anlamdalar ama önemli bir bilgi farklı ya da eksik.",
        "4: Büyük ölçüde aynı anlamdalar; yalnızca önemsiz ayrıntılar farklı.",
        "5: Tamamen aynı anlama geliyorlar.",
    ],
}


def to_items(filename):
    items = []
    for r in read_tsv(download(BASE + filename).read_text(encoding="utf-8")):
        score = float(r["score"])
        items.append({
            "state": {"cümle_1": r["sentence1"], "cümle_2": r["sentence2"]},
            "questions": {"similarity": QUESTION},
            "gold": {"similarity": int(score + 0.5)},
            "meta": {"score": score},
        })
    return items


def build():
    return split(to_items("stsb_tr_test.tsv"), to_items("stsb_tr_dev.tsv"))
