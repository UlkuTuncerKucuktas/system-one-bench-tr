import re

from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "noul",
    "instructions": "Bu haber metnini bir yapay zekâ modeli mi yazdı?",
    "criteria": {"true": "Metni bir dil modeli üretti.", "false": "Metni bir gazeteci yazdı."},
}
PROMPT = re.compile(r"(?:\[INST\] )?Write a news article in Turkish\. Headline: (.*?) Article: (.*?) ?(?:\[EOP\]|\[/INST\])", re.S)


def to_items(split_name):
    return [
        {
            "state": PROMPT.sub(r"\1\n\2", r["body"]).replace("[EOP]", ""),
            "questions": {"generated": QUESTION},
            "gold": {"generated": r["label"] == "neural"},
            "meta": {"model": r["model"]},
        }
        for r in hf_rows("tum-nlp/neural-news-benchmark", None, split_name)
        if r["language"] == "tr"
    ]


def build():
    return split(to_items("test"), to_items("validation"), to_items("train"))
