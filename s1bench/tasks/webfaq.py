import re

from ..download import hf_rows
from ..split import split

REPO = "mteb/WebFAQRetrieval"
QUESTION = {
    "type": "noul",
    "instructions": "Bu cevap soruyu yanıtlıyor mu?",
    "criteria": {"true": "Cevap sorulan şeyi yanıtlıyor.", "false": "Cevap başka bir soruya ait ya da sorulanı yanıtlamıyor."},
}


def build():
    queries = {r["id"]: r["text"] for r in hf_rows(REPO, "tur-queries", "test")}
    corpus = {r["id"]: r["text"] for r in hf_rows(REPO, "tur-corpus", "test")}
    answers = {r["query-id"]: corpus[r["corpus-id"]] for r in hf_rows(REPO, "tur-qrels", "test")}
    words = {i: set(re.findall(r"\w+", queries[i].lower())) for i in answers}

    def similar_question(i):
        # the closest other question that is not a near-copy of this one, since a near-copy's answer may be right too
        others = (j for j in answers if len(words[i] & words[j]) < len(words[i] | words[j]) / 2)
        return max(others, key=lambda j: len(words[i] & words[j]))

    items = []
    for n, i in enumerate(sorted(answers)):
        answer, fits = (answers[similar_question(i)], False) if n % 2 else (answers[i], True)
        items.append({"state": {"soru": queries[i], "cevap": answer}, "questions": {"fits": QUESTION}, "gold": {"fits": fits}})
    return split(items)
