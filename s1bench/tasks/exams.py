import re

from ..download import hf_rows
from ..split import shuffled, split

INSTRUCTIONS = "Bu sorunun doğru cevabı hangisidir?"
DIFFICULTY = {
    "type": "score",
    "instructions": "Bu soruyu öğrencilerin ne kadarı doğru cevapladı?",
    "criteria": [
        "Kolay: öğrencilerin en az %42'si doğru cevapladı",
        "Orta: öğrencilerin %28 ile %41'i doğru cevapladı",
        "Zor: öğrencilerin en fazla %27'si doğru cevapladı",
    ],
}
LEVELS = ["easy", "medium", "hard"]


def exam_item(r, answer):
    options = dict(zip("ABCD", [r["option_a"], r["option_b"], r["option_c"], r["option_d"]]))
    return {
        "state": r["question"],
        "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}},
        "gold": {"answer": answer},
    }


def normalised(text):
    return " ".join(re.findall(r"\w+", text.lower()))


def include_items(split_name):
    # INCLUDE repeats 137 TurkishMMLU questions with one option dropped, often the right one, and answers that disagree with
    # TurkishMMLU's; turkishmmlu_tr asks them properly
    turkishmmlu = {normalised(r["question"]) for r in hf_rows("AYueksel/TurkishMMLU", "All", "test")}
    rows = hf_rows("CohereLabs/include-base-44", "Turkish", split_name)
    return [exam_item(r, "ABCD"[r["answer"]]) for r in rows if normalised(r["question"]) not in turkishmmlu]


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


def build_turkishmmlu():
    items = []
    for r in hf_rows("AYueksel/TurkishMMLU", "All", "test"):
        options = dict(zip("ABCDE", r["choices"]))
        items.append({
            "state": r["question"],
            "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}},
            "gold": {"answer": "ABCDE"[r["answer"]]},
            "meta": {"subject": r["subject"], "grade": r["metadata"]["grade"]},
        })
    return split(items)


def build_turkishmmlu_difficulty():
    items = []
    for r in hf_rows("AYueksel/TurkishMMLU", "All", "test"):
        state = {"sınıf": r["metadata"]["grade"], "soru": r["question"], "seçenekler": dict(zip("ABCDE", r["choices"])), "doğru_cevap": "ABCDE"[r["answer"]]}
        items.append({"state": state, "questions": {"difficulty": DIFFICULTY}, "gold": {"difficulty": LEVELS.index(r["metadata"]["difficulty"])}})
    return split(items)


def build_tus21():
    items = []
    for r in hf_rows("zypchn/TUS21-exams", None, "train"):
        options = {choice[0]: choice[3:] for choice in r["choices"]}
        items.append({
            "state": r["question"],
            "questions": {"answer": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": options}},
            "gold": {"answer": r["answer_idx"]},
        })
    return split(items)
