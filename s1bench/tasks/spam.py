import csv

from ..download import kaggle_file
from ..split import split

QUESTION = {
    "type": "noul",
    "instructions": "Bu SMS istenmeyen (spam) bir mesaj mı?",
    "criteria": {
        "true": "Reklam, kampanya, dolandırıcılık ya da toplu gönderilmiş istenmeyen bir mesaj.",
        "false": "Birine yazılmış normal bir mesaj.",
    },
}


def build():
    path = kaggle_file("onurkarasoy/turkish-sms-collection", "TurkishSMSCollection.csv")
    rows = csv.DictReader(open(path, encoding="utf-8", newline=""), delimiter=";")
    return split([{"state": r["Message"], "questions": {"spam": QUESTION}, "gold": {"spam": r["Group"] == "1"}} for r in rows])
