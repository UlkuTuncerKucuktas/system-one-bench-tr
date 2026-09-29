from ..download import hf_rows
from ..split import shuffled, split

INSTRUCTIONS = "Bu sorunun doğru cevabı hangisidir?"


def exam_item(r, answer):
    options = dict(zip("ABCD", [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]))
    return {
        "state": r["question"],
        "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}},
        "gold": {"answer": answer},
    }


def include_items(split_name):
    return [exam_item(r, "ABCD"[r["answer"]]) for r in hf_rows("CohereLabs/include-base-44", "Turkish", split_name)]


def build_include():
    return split(include_items("test"), include_items("validation"))


def global_mmlu_items(split_name):
    items = []
    for r in hf_rows("CohereLabs/Global-MMLU", "tr", split_name):
        item = exam_item(r, r["answer"])
        item["meta"] = {"cultural_sensitivity": r["cultural_sensitivity_label"], "subject_category": r["subject_category"]}
        items.append(item)
    return items


def build_global_mmlu():
    test = global_mmlu_items("test")
    sensitive = shuffled(item for item in test if item["meta"]["cultural_sensitivity"] == "CS")
    agnostic = shuffled(item for item in test if item["meta"]["cultural_sensitivity"] == "CA")
    unlabelled = [item for item in test if item["meta"]["cultural_sensitivity"] == "-"]
    return split(sensitive[:500] + agnostic[:500], global_mmlu_items("dev"), sensitive[500:] + agnostic[500:] + unlabelled)
