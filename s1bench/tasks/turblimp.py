from ..download import hf_rows
from ..split import split

PHENOMENA = [
    "anaphor_agreement",
    "argument_structure_ditransitive",
    "argument_structure_transitive",
    "binding",
    "determiners",
    "ellipsis",
    "irregular_forms",
    "island_effects",
    "nominalization",
    "npi_licensing",
    "passives",
    "quantifiers",
    "relative_clauses",
    "scrambling",
    "subject_agreement",
    "suspended_affixation",
]
QUESTION = {
    "type": "choice",
    "instructions": "Bu iki cümleden hangisi dilbilgisi açısından doğru?",
    "criteria": {"birinci": "Birinci cümle", "ikinci": "İkinci cümle"},
}


def build():
    items = []
    for phenomenon in PHENOMENA:
        for i, r in enumerate(hf_rows("juletxara/turblimp", phenomenon, "train")):
            if r["sentence_good"] != r["sentence_bad"]:
                good_first = i % 2 == 0
                first, second = (r["sentence_good"], r["sentence_bad"]) if good_first else (r["sentence_bad"], r["sentence_good"])
                items.append({
                    "state": {"birinci": first, "ikinci": second},
                    "questions": {"grammatical": QUESTION},
                    "gold": {"grammatical": "birinci" if good_first else "ikinci"},
                    "meta": {"phenomenon": phenomenon},
                })
    return split(items)
