from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "noul",
    "instructions": "Anayasa Mahkemesi bu bireysel başvuruda en az bir hakkın ihlal edildiğine karar vermiş midir?",
    "criteria": {
        "true": "Mahkeme en az bir hakkın ihlal edildiğine karar verdi.",
        "false": "Mahkeme ihlal bulmadı ya da başvuruyu kabul edilemez buldu.",
    },
}


def to_items(split_name):
    return [
        {
            "state": {"haklar": r["Haklar"], "başvuru": r["text"][:8000]},
            "questions": {"violation": QUESTION},
            "gold": {"violation": r["labels"] == 1},
        }
        for r in hf_rows("icgcihan/Turkish_Constutional_Court_Decisions", None, split_name)
    ]


def build():
    return split(to_items("test"), to_items("validation"), to_items("train"))
