from ..download import hf_rows
from ..split import split

QUESTION = {
    "type": "noul",
    "instructions": "Bu cümlede ironi var mı?",
    "criteria": {"true": "Söylenenden farklı, çoğu zaman tersi bir şey kastediliyor.", "false": "Söylenen düz anlamıyla kastediliyor."},
}


def build():
    items = [
        {"state": r["text"], "questions": {"ironic": QUESTION}, "gold": {"ironic": r["label"] == 1}}
        for r in hf_rows("mcemilg/IronyTR", None, "validation")
    ]
    return split(items)
