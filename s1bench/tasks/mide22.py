import re

from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "choice",
    "instructions": "Bu tweet doğru bilgi mi, yanlış bilgi mi yayıyor?",
    "criteria": {
        "doğru": "Tweet doğru bilgi içeriyor.",
        "yanlış": "Tweet yanlış ya da yanıltıcı bilgi yayıyor.",
        "diğer": "Tweet doğru ya da yanlış diye değerlendirilebilecek bir bilgi içermiyor.",
    },
}
LABELS = {"True": "doğru", "False": "yanlış", "Other": "diğer"}


def build():
    items = [
        {"state": re.sub(r"@\w+", "@USER", r["tweet"]), "questions": {"verdict": QUESTION}, "gold": {"verdict": LABELS[r["label"]]}}
        for r in hf_rows("ogozcelik/turkish-fake-news-detection", None, "train")
        if r["label"]
    ]
    return split(items)
