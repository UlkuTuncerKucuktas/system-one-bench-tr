from datasets import load_dataset

from ..download import hf_rows
from ..split import split

SUPPORTED = {
    "type": "noul",
    "instructions": "Cevaptaki her bilgi verilen metne dayanıyor mu?",
    "criteria": {
        "true": "Cevaptaki her bilgi metinde yer alıyor ya da metinden çıkıyor.",
        "false": "Cevapta metinde olmayan ya da metinle çelişen bilgi var.",
    },
}
BETTER = {
    "type": "choice",
    "instructions": "Hangi yanıt isteği daha iyi karşılıyor?",
    "criteria": {"birinci": "Birinci yanıt daha iyi.", "ikinci": "İkinci yanıt daha iyi."},
}
QUALITY = {
    "type": "score",
    "instructions": "Bu Türkçe çeviri, kaynak metnin anlamını ne kadar iyi aktarıyor?",
    "criteria": [
        "0–19: Anlam büyük ölçüde kaybolmuş",
        "20–39: Anlamın çoğu kaybolmuş",
        "40–59: Anlamın bir kısmı aktarılmış",
        "60–79: Anlamın çoğu aktarılmış",
        "80–100: Anlam tamamen aktarılmış",
    ],
}


def build_ragtruth():
    items = {"train": [], "test": []}
    for r in hf_rows("newmindai/RAGTruth-TR", None, "train"):
        items[r["split"]].append({
            "state": {"istem": r["prompt"], "cevap": r["answer"]},
            "questions": {"supported": SUPPORTED},
            "gold": {"supported": not r["labels"]},
            "meta": {"task_type": r["task_type"]},
        })
    return split(items["test"], rest=items["train"])


def build_rewardbench():
    items = []
    for i, r in enumerate(hf_rows("CohereLabsCommunity/multilingual-reward-bench", "tur_Latn", "test")):
        chosen_first = i % 2 == 0
        first, second = (r["chosen"], r["rejected"]) if chosen_first else (r["rejected"], r["chosen"])
        items.append({
            "state": {"istek": r["prompt"], "birinci_yanıt": first, "ikinci_yanıt": second},
            "questions": {"better": BETTER},
            "gold": {"better": "birinci" if chosen_first else "ikinci"},
            "meta": {"category": r["category"]},
        })
    return split(items)


def build_wmt():
    rows = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train").filter(lambda r: r["lp"] == "en-tr")
    items = [
        {"state": {"kaynak": r["src"], "çeviri": r["mt"]}, "questions": {"quality": QUALITY}, "gold": {"quality": min(int(r["raw"] // 20), 4)}}
        for r in rows
    ]
    return split(items)
