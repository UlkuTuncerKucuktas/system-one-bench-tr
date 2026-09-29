import json

from ..download import hf_rows
from ..split import split

QUESTIONS = {
    "The summary can be read and understood by the rater.": ("comprehensible", {
        "type": "noul",
        "instructions": "Özet okunup anlaşılabiliyor mu?",
        "criteria": {"true": "Özet okunabiliyor ve ne anlattığı anlaşılıyor.", "false": "Özet anlaşılmıyor, bozuk ya da okunamıyor."},
    }),
    "The summary is free of unnecessarily repeated information.": ("repetition", {
        "type": "noul",
        "instructions": "Özet gereksiz yere tekrarlanan bilgiden arınmış mı?",
        "criteria": {"true": "Özette gereksiz tekrar yok.", "false": "Özet aynı bilgiyi gereksiz yere tekrarlıyor."},
    }),
    "The summary is grammatically correct.": ("grammar", {
        "type": "noul",
        "instructions": "Özet dilbilgisi açısından doğru mu?",
        "criteria": {"true": "Özette dilbilgisi hatası yok.", "false": "Özette dilbilgisi hataları var."},
    }),
    "All the information in the summary is fully attributable to the source article.": ("attribution", {
        "type": "noul",
        "instructions": "Özetteki bilgilerin tamamı makaleye dayanıyor mu?",
        "criteria": {
            "true": "Özetteki her bilgi makalede yer alıyor ya da makaleden açıkça çıkıyor.",
            "false": "Özette makalede olmayan ya da makaleyle çelişen bilgi var.",
        },
    }),
    "The summary captures the main idea(s) of the source article.": ("main_ideas", {
        "type": "noul",
        "instructions": "Özet makalenin ana fikrini veriyor mu?",
        "criteria": {"true": "Özet makalenin ana fikrini ya da fikirlerini yakalıyor.", "false": "Özet makalenin ana fikrini kaçırıyor."},
    }),
    "The summary concisely represents the information in the source article.": ("conciseness", {
        "type": "noul",
        "instructions": "Özet makaledeki bilgiyi kısa ve öz biçimde veriyor mu?",
        "criteria": {"true": "Özet bilgiyi gereksiz ayrıntıya girmeden veriyor.", "false": "Özet gereksiz ayrıntı içeriyor ya da özlü değil."},
    }),
}


def to_items(split_name):
    items = {}
    for r in hf_rows("tasksource/seahorse_summarization_evaluation", None, split_name):
        if r["worker_lang"] == "tr":
            qid, question = QUESTIONS[r["question"]]
            summary = json.loads(f'"{r["summary"]}"')
            item = items.setdefault((r["gem_id"], r["model"]), {"state": {"makale": r["article"], "özet": summary}, "questions": {}, "gold": {}})
            item["questions"][qid] = question
            item["gold"][qid] = r["answer"] == "Yes"
    return list(items.values())


def build():
    return split(to_items("test"), to_items("validation"), to_items("train"))
