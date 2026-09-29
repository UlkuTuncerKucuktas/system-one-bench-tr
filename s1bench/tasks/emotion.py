import csv
import re

from ..download import hf_file
from ..split import split

LABELS = {"kizgin": "kızgın", "korku": "korkmuş", "mutlu": "mutlu", "surpriz": "şaşırmış", "uzgun": "üzgün"}
QUESTION = {"type": "choice", "instructions": "Bu tweeti yazan kişi nasıl hissediyor?", "criteria": dict.fromkeys(LABELS.values())}


def build():
    path = hf_file("anilguven/turkish_tweet_emotion_dataset", "Turkish_Tweet_Dataset.csv")
    items = [
        {"state": re.sub(r"@\w+", "@USER", text.strip()), "questions": {"emotion": QUESTION}, "gold": {"emotion": LABELS[label]}}
        for text, label in csv.reader(open(path, encoding="utf-8"))
    ]
    return split(items)
