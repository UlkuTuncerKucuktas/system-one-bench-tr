import re

from ..download import download, hf_rows, read_tsv, read_zip
from ..split import split

BASE = "https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab/-/raw/main/task1/data/subtasks-turkish/"
CHECKWORTHY = {
    "type": "noul",
    "instructions": "Bu tweet doğruluk kontrolüne değer mi? Kamuoyunu ilgilendiren ve bir doğruluk kontrolcüsünün incelemesi gereken bir iddia içeriyor mu?",
    "criteria": {"true": "Kamuoyunu ilgilendiren ve doğrulanması gereken bir iddia içeriyor.", "false": "Doğruluk kontrolüne değer bir iddia içermiyor."},
}
CLAIM = {
    "type": "noul",
    "instructions": "Bu tweet doğruluğu kontrol edilebilecek olgusal bir iddia içeriyor mu?",
    "criteria": {
        "true": "Doğru ya da yanlış olduğu kontrol edilebilecek olgusal bir iddia içeriyor.",
        "false": "Görüş, soru, şaka ya da kontrol edilemeyecek bir ifade içeriyor.",
    },
}
HARMFUL = {
    "type": "noul",
    "instructions": "Bu tweet topluma zarar verebilir mi?",
    "criteria": {
        "true": "Yanlış bilgi, panik, nefret ya da hedef gösterme gibi zarar verebilecek bir içerik taşıyor.",
        "false": "Zarar verebilecek bir içerik taşımıyor.",
    },
}
SUBTASKS = {"checkworthy": ("1A_checkworthy", CHECKWORTHY), "claim": ("1B_claim", CLAIM), "harmful": ("1C_harmful", HARMFUL)}


def tweet(text):
    return re.sub(r"@\w+", "@USER", text)


def checkthat_items(part):
    items = {}
    for qid, (subtask, question) in SUBTASKS.items():
        name = f"CT22_turkish_{subtask}"
        url = BASE + (f"test/{name}_test_gold.zip" if part == "test_gold" else f"{name}.zip")
        for r in read_tsv(read_zip(download(url), f"{name}_{part}.tsv")):
            item = items.setdefault(r["tweet_id"], {"state": tweet(r["tweet_text"]), "questions": {}, "gold": {}})
            item["questions"][qid] = question
            item["gold"][qid] = r["class_label"] == "1"
    return list(items.values())


def build_checkthat():
    return split(checkthat_items("test_gold") + checkthat_items("dev_test"), checkthat_items("dev"), checkthat_items("train"))


def build_trclaim19():
    items = {}
    for r in hf_rows("mcemilg/TrClaim19", None, "train"):
        votes = r["graded_check_wortiness"] / 3
        items[r["tweet"]] = {
            "state": tweet(r["tweet"]),
            "questions": {"checkworthy": CHECKWORTHY},
            "gold": {"checkworthy": r["check_worthiness"] == 1},
            "soft_gold": {"checkworthy": {"true": votes, "false": 1 - votes}},
        }
    return split(items.values())
