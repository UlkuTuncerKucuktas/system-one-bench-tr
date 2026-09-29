import json

from ..download import download, read_zip
from ..split import split

URL = "https://nli-tr.s3.eu-central-1.amazonaws.com/multinli_tr_1.0.zip"
RELATIONS = {
    "entailment": "Öncül doğruysa hipotez de kesinlikle doğrudur.",
    "neutral": "Öncül, hipotezin doğru mu yanlış mı olduğunu belirlemez.",
    "contradiction": "Öncül doğruysa hipotez kesinlikle yanlıştır.",
}
QUESTION = {
    "type": "choice",
    "instructions": "Öncül doğru kabul edilirse hipotez hakkında ne söylenebilir?",
    "criteria": RELATIONS,
}


def to_items(filename):
    items = []
    for line in read_zip(download(URL), filename).splitlines():
        r = json.loads(line)
        if r["gold_label"] == "-":
            continue
        votes = r["annotator_labels"]
        items.append({
            "state": {"öncül": r["sentence1"], "hipotez": r["sentence2"]},
            "questions": {"relation": QUESTION},
            "gold": {"relation": r["gold_label"]},
            "soft_gold": {"relation": {relation: votes.count(relation) / len(votes) for relation in RELATIONS}},
        })
    return items


def build():
    return split(to_items("multinli_tr_1.0_dev_matched.jsonl"), to_items("multinli_tr_1.0_dev_mismatched.jsonl"))
