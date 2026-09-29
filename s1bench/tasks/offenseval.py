from ..download import download, read_tsv, read_zip
from ..split import split

URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
QUESTION = {
    "type": "noul",
    "instructions": "Bu tweet saldırgan (offensive) bir ifade içeriyor mu?",
    "criteria": {
        "true": "Hakaret, küfür, tehdit, aşağılama ya da hedef gösterme içeriyor.",
        "false": "Saldırgan bir ifade içermiyor.",
    },
}


def to_items(tweets, labels):
    return [
        {"state": t["tweet"], "questions": {"offensive": QUESTION}, "gold": {"offensive": labels[t["id"]] == "OFF"}}
        for t in tweets
    ]


def build():
    path = download(URL)
    train = read_tsv(read_zip(path, "offenseval-tr-training-v1.tsv"))
    test = read_tsv(read_zip(path, "offenseval-tr-testset-v1.tsv"))
    test_labels = read_tsv(read_zip(path, "offenseval-tr-labela-v1.tsv"), delimiter=",", fieldnames=["id", "label"])
    return split(
        to_items(test, {r["id"]: r["label"] for r in test_labels}),
        to_items(train, {r["id"]: r["subtask_a"] for r in train}),
    )
