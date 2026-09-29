from ..download import download, read_tsv
from ..split import split

BASE = "https://raw.githubusercontent.com/utahnlp/x-fact/main/data/x-fact/"
INSTRUCTIONS = "Bu iddianın doğruluk durumu nedir? Kanıtlar web aramasından alınmış kısa parçalardır ve eksik olabilir."
VERDICTS = {
    "true": "doğru",
    "mostly true": "çoğunlukla doğru",
    "partly true/misleading": "kısmen doğru / yanıltıcı",
    "mostly false": "çoğunlukla yanlış",
    "false": "yanlış",
    "complicated/hard to categorise": "belirsiz",
}
FACT_CHECKERS = ["dogrulukpayi", "doğruluk pay", "teyit.org", "teyitorg", "malumatfurus", "yalansavar"]


def turkish_rows(filename):
    return [r for r in read_tsv(download(BASE + filename).read_text(encoding="utf-8")) if r["language"] == "tr"]


def evidence(r):
    snippets = []
    for i in range(1, 6):
        text = r[f"evidence_{i}"]
        seen_on = (text + " " + r[f"link_{i}"]).lower()
        if text and not any(name in seen_on for name in FACT_CHECKERS):
            snippets.append(text)
    return snippets


def to_items(rows, verdicts):
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict.fromkeys(verdicts)}
    return [
        {
            "state": {"iddia": r["claim"], "iddia_sahibi": r["claimant"], "kanitlar": evidence(r)},
            "questions": {"verdict": question},
            "gold": {"verdict": VERDICTS[r["label"]]},
        }
        for r in rows
    ]


def build():
    verdicts = ["doğru", "çoğunlukla doğru", "kısmen doğru / yanıltıcı", "çoğunlukla yanlış", "yanlış"]
    return split(to_items(turkish_rows("test.all.tsv"), verdicts), to_items(turkish_rows("dev.all.tsv"), verdicts))


def build_teyit():
    verdicts = ["doğru", "kısmen doğru / yanıltıcı", "yanlış", "belirsiz"]
    return split(to_items(turkish_rows("ood.tsv"), verdicts))
