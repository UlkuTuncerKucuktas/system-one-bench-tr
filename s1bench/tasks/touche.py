from ..download import download, read_tsv, read_zip
from ..split import split

URL = "https://zenodo.org/records/10450641/files/trainingset-ideology-power.zip"
POWER = {
    "type": "noul",
    "instructions": "Bu konuşmayı yapan milletvekilinin partisi iktidarda mı?",
    "criteria": {"true": "Konuşmacının partisi iktidarda ya da iktidar ortağı.", "false": "Konuşmacının partisi muhalefette."},
}
ORIENTATION = {
    "type": "choice",
    "instructions": "Bu konuşmayı yapan milletvekilinin partisi siyasi yelpazenin neresinde?",
    "criteria": {"sol": "Solda ya da sol eğilimli", "sağ": "Sağda ya da sağ eğilimli"},
}


def build():
    path = download(URL)
    items = {}
    for r in read_tsv(read_zip(path, "power-tr-train.tsv")):
        item = items.setdefault(r["text"], {"state": r["text"][:8000], "questions": {}, "gold": {}})
        item["questions"]["in_power"] = POWER
        item["gold"]["in_power"] = r["label"] == "0"
    for r in read_tsv(read_zip(path, "orientation-tr-train.tsv")):
        item = items.setdefault(r["text"], {"state": r["text"][:8000], "questions": {}, "gold": {}})
        item["questions"]["orientation"] = ORIENTATION
        item["gold"]["orientation"] = "sağ" if r["label"] == "1" else "sol"
    return split(items.values())
