from . import belebele, exams, jev_noul, massive, nli, offenseval, reviews, stsb, xcopa, xfact

TASKS = {
    "massive_intent_tr": massive.build,
    "belebele_tr": belebele.build,
    "xcopa_tr": xcopa.build,
    "include_tr": exams.build_include,
    "global_mmlu_tr": exams.build_global_mmlu,
    "nli_tr": nli.build,
    "xfact_tr": xfact.build,
    "xfact_teyit_tr": xfact.build_teyit,
    "offenseval_tr": offenseval.build,
    "jev_noul_tr": jev_noul.build,
    "jev_noul_unseen_tr": jev_noul.build_unseen,
    "buyuksinema_tr": reviews.build_buyuksinema,
    "musteri_yorumlari_tr": reviews.build_musteri_yorumlari,
    "stsb_tr": stsb.build,
}
