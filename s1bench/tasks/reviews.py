from ..download import hf_rows
from ..split import split

MOVIE_QUESTION = {
    "type": "score",
    "instructions": "Bu yorumu yazan kişi filme 10 üzerinden kaç puan vermiştir?",
    "criteria": [
        "1/10: En düşük puan, hiç beğenmemiş",
        "2/10: İzlemeye değmez",
        "3/10: Berbat",
        "4/10: Zayıf",
        "5/10: Kötü, ortalamanın altında",
        "6/10: Orta",
        "7/10: İyi",
        "8/10: Çok iyi",
        "9/10: Harika",
        "10/10: En yüksek puan, bayılmış",
    ],
}
PRODUCT_QUESTION = {
    "type": "score",
    "instructions": "Bu yorumu yazan müşteri ürüne kaç yıldız vermiştir?",
    "criteria": [
        "1 yıldız: Çok olumsuz",
        "2 yıldız: Olumsuz",
        "3 yıldız: Ne iyi ne kötü",
        "4 yıldız: Olumlu",
        "5 yıldız: Çok olumlu",
    ],
}


def to_items(repo, split_name, question):
    return [
        {"state": r["text"], "questions": {"rating": question}, "gold": {"rating": r["label"]}}
        for r in hf_rows(repo, None, split_name)
    ]


def build_reviews(repo, question):
    return split(to_items(repo, "test", question), to_items(repo, "validation", question), to_items(repo, "train", question))


def build_buyuksinema():
    return build_reviews("turkish-nlp-suite/BuyukSinema", MOVIE_QUESTION)


def build_musteri_yorumlari():
    return build_reviews("turkish-nlp-suite/MusteriYorumlari", PRODUCT_QUESTION)
