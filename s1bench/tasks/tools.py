from ..download import hf_rows
from ..split import split

INSTRUCTIONS = "Kullanıcının isteğini karşılamak için hangi araç çağrılmalı?"


def build():
    rows = hf_rows("BTX24/turkish-mobile-function-calling-dataset", None, "train")
    question = {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict.fromkeys(sorted({r["tool_name"] for r in rows}))}
    return split([{"state": r["user_content"], "questions": {"tool": question}, "gold": {"tool": r["tool_name"]}} for r in rows])
