import re

from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "choice",
    "instructions": "Bu tweet'in duygu tonu nedir?",
    "criteria": {
        "olumsuz": "Öfke, eleştiri, üzüntü ya da hoşnutsuzluk taşıyor.",
        "nötr": "Bilgi veriyor ya da belirgin bir duygu taşımıyor.",
        "olumlu": "Destek, sevinç ya da övgü taşıyor.",
        "alaycı": "İğneleyici ya da ironik; söylenenin tersi kastediliyor.",
    },
}
LABELS = {"Negative": "olumsuz", "Neutral": "nötr", "Positive": "olumlu", "Sarcastic": "alaycı"}


def build():
    items = []
    for r in hf_rows("ctoraman/BilTweetNews-sentiment-analysis", None, "train"):
        if r["Majority"] in LABELS:
            total = sum(r[label] for label in LABELS)
            items.append({
                "state": re.sub(r"@\w+", "@USER", r["Text"]),
                "questions": {"sentiment": QUESTION},
                "gold": {"sentiment": LABELS[r["Majority"]]},
                "soft_gold": {"sentiment": {LABELS[label]: r[label] / total for label in LABELS}},
            })
    return split(items)
