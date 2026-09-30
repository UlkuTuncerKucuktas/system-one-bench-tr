# system-one-bench-tr

Turkish decision tasks for comparing System One models (TypeSafe Jev and open look-alikes) with
ordinary LLMs on exactly the same questions.

Every item is a TypeSafe System One request (`state` + `questions`) with its gold answer. A System
One model answers it directly; an LLM answers it through an adapter.

## Build

```bash
pip install -r requirements.txt
python -m s1bench.build
```

This writes `data/<task>/dev.jsonl`, `data/<task>/test.jsonl` and `data/<task>/rest.jsonl`. To
rebuild some tasks only: `python -m s1bench.build nli_tr xcopa_tr`.

`rest.jsonl` holds the rows of a source that are in neither dev nor test, its train split included.
Nothing evaluates it; it is there so that training data can be kept away from every benchmark source
(see below).

HateDay is gated: accept its terms on Hugging Face and log in with `hf auth login` before building
hateday_tr. The Kaggle sources download without an account.

## Evaluate

```bash
python -m s1bench.evaluate qwen3.5-4b
python -m s1bench.score
```

`evaluate` runs one model on every task (or on the tasks you list after the model name) and writes
its probabilities to `results/<model>/<task>/`. `score` fits one temperature per model, task and
question on dev, then writes one row per question to `results/scores.csv`: accuracy, Brier score,
calibration error (ECE), for Score questions the mean absolute error of the expected level, and,
where annotators' votes are known, `vote_distance`, the total variation distance between the model's
probabilities and the votes. A question with fewer than 30 dev items (many in bev_tr) gets the
temperature fitted on the whole task. `raw_nll`, `raw_brier` and `raw_ece` measure the model's own
probabilities before any temperature, which is how well a model is calibrated on questions it has no
dev items for. Macro-F1 averages the classes that occur in test. `score` also prints accuracy averaged
over the tasks of each area, where a task's accuracy weights its questions by their number of items;
the secondary tasks are left out of those averages.

Models are listed in `s1bench/models.py`:

- Ordinary LLMs see the state, the question and lettered options, and we read the probability of
  each letter as the next token in one forward pass, with thinking turned off. Choice and Noul
  questions are asked twice with the options in reverse order and the two answers are averaged.
  The letters are A–Z, a–z and 0–9, so a question with more than 62 options is skipped (banking77,
  clinc150 and ledgar in jev-bench-tr).
- Metask-Jev-4B uses its own prompt format and handles at most 26 options, so besides those it skips
  MASSIVE and jev-bench-tr's MASSIVE and GoEmotions. It reads prompts of up to 32,768 tokens, since
  the other models read the whole state too.
- Laya uses its multilingual checkpoint with `max_len=8192`; by default it cuts documents at 1,024 tokens.
- Jev needs your key in `TYPESAFE_API_KEY`.
- Jev and Laya get one question per request, like the LLMs. Sending an item's questions together
  would give labels away: seahorse_tr asks its other five questions only when the raters understood
  the summary.

## Keeping training data out of the benchmark

```bash
python -m s1bench.fingerprints
```

This writes `data/fingerprints.txt`: a hash of every 8-word window of every dev and test text, and a
hash of every whole text in `rest.jsonl`. Models we train never see a text that copies the benchmark:
one whose whole text matches a hash in that file (for texts of four words or more), half of whose
8-word windows are in it, or that has 10 windows in it in a row (17 words) making up a fifth of the
text. Each text of an item is checked, and so are all of them joined, since a benchmark text can be
split over several fields. Scattered shared windows and runs inside long texts do not count, because
they are common phrases and quoted laws ("5271 sayılı Ceza Muhakemesi Kanunu"), and very short texts
such as "tamam" appear everywhere.

## Results

Accuracy, macro-F1, Brier score and ECE average the 25 area tasks that every model answered (all but
MASSIVE, which Metask skips); a task's value weights its questions by their number of items. ECE is
measured after each question's dev temperature, raw ECE on the model's own probabilities. Score MAE
averages the three Score tasks. The last two columns average the 18 secondary tasks outside
jev-bench-tr and the 17 jev-bench-tr tasks that every LLM and Metask answered; Jev has results for only
three of the secondary tasks. The last row always answers the most common label of each question's
test items.

| Model | Accuracy | Macro-F1 | Brier | ECE | Raw ECE | MASSIVE accuracy | Score MAE | Secondary accuracy | jev-bench-tr accuracy |
|---|---|---|---|---|---|---|---|---|---|
| `jev` | 0.776 | 0.748 | 0.300 | 0.063 | 0.085 | 0.788 | 0.86 | – | – |
| `gemma-4-12b` | 0.759 | 0.728 | 0.326 | 0.054 | 0.192 | 0.777 | 0.82 | 0.678 | 0.637 |
| `qwen3.5-35b-a3b` | 0.741 | 0.701 | 0.333 | 0.052 | 0.119 | 0.771 | 0.89 | 0.677 | 0.655 |
| `qwen3.5-9b` | 0.706 | 0.665 | 0.375 | 0.057 | 0.124 | 0.691 | 1.03 | 0.630 | 0.626 |
| `trendyol-asure-12b` | 0.702 | 0.666 | 0.365 | 0.069 | 0.140 | 0.685 | 0.89 | 0.604 | 0.579 |
| `metask-jev-4b` | 0.697 | 0.654 | 0.391 | 0.067 | 0.201 | – | 0.98 | 0.607 | 0.654 |
| `gemma-4-e4b` | 0.689 | 0.659 | 0.385 | 0.061 | 0.224 | 0.720 | 0.97 | 0.591 | 0.579 |
| `turkish-gemma-9b` | 0.662 | 0.630 | 0.406 | 0.085 | 0.232 | 0.652 | 1.12 | 0.586 | 0.547 |
| `qwen3.5-4b` | 0.659 | 0.615 | 0.428 | 0.059 | 0.134 | 0.570 | 1.35 | 0.582 | 0.623 |
| `eurollm-9b` | 0.614 | 0.550 | 0.457 | 0.082 | 0.172 | 0.565 | 1.45 | 0.542 | 0.527 |
| `kizagan-e4b` | 0.603 | 0.562 | 0.480 | 0.063 | 0.179 | 0.514 | 1.46 | 0.524 | 0.496 |
| `laya-multilingual` | 0.459 | 0.413 | 0.592 | 0.062 | 0.254 | 0.347 | 1.67 | 0.424 | 0.417 |
| `mecellem-qwen3-4b` | 0.370 | 0.293 | 0.651 | 0.064 | 0.108 | 0.006 | 1.67 | 0.357 | 0.375 |
| `kumru-2b` | 0.360 | 0.278 | 0.647 | 0.057 | 0.084 | 0.003 | 1.69 | 0.393 | 0.365 |
| always the most common label | 0.446 | – | – | – | – | 0.062 | – | 0.462 | 0.493 |

Accuracy by area, over all the tasks of each area:

| Model | routing | opinion | safety | fact_checking | judging | workflow | politics | language | knowledge |
|---|---|---|---|---|---|---|---|---|---|
| `jev` | 0.874 | 0.624 | 0.906 | 0.512 | 0.817 | 0.873 | 0.868 | 0.789 | 0.869 |
| `gemma-4-12b` | 0.862 | 0.614 | 0.865 | 0.585 | 0.820 | 0.873 | 0.853 | 0.778 | 0.742 |
| `qwen3.5-35b-a3b` | 0.858 | 0.598 | 0.822 | 0.529 | 0.762 | 0.860 | 0.860 | 0.759 | 0.793 |
| `qwen3.5-9b` | 0.811 | 0.545 | 0.806 | 0.564 | 0.777 | 0.842 | 0.767 | 0.701 | 0.705 |
| `trendyol-asure-12b` | 0.831 | 0.621 | 0.716 | 0.397 | 0.778 | 0.871 | 0.829 | 0.753 | 0.675 |
| `metask-jev-4b` | – | 0.536 | 0.914 | 0.551 | 0.748 | 0.789 | 0.744 | 0.681 | 0.621 |
| `gemma-4-e4b` | 0.832 | 0.591 | 0.817 | 0.444 | 0.720 | 0.861 | 0.802 | 0.691 | 0.630 |
| `turkish-gemma-9b` | 0.813 | 0.584 | 0.557 | 0.423 | 0.729 | 0.860 | 0.865 | 0.688 | 0.672 |
| `qwen3.5-4b` | 0.759 | 0.499 | 0.677 | 0.521 | 0.789 | 0.829 | 0.641 | 0.679 | 0.638 |
| `eurollm-9b` | 0.750 | 0.520 | 0.712 | 0.430 | 0.651 | 0.842 | 0.736 | 0.601 | 0.479 |
| `kizagan-e4b` | 0.739 | 0.463 | 0.712 | 0.388 | 0.726 | 0.762 | 0.653 | 0.639 | 0.462 |
| `laya-multilingual` | 0.567 | 0.303 | 0.539 | 0.431 | 0.594 | 0.675 | 0.492 | 0.446 | 0.243 |
| `mecellem-qwen3-4b` | 0.088 | 0.160 | 0.385 | 0.478 | 0.571 | 0.550 | 0.486 | 0.420 | 0.309 |
| `kumru-2b` | 0.065 | 0.158 | 0.643 | 0.441 | 0.510 | 0.515 | 0.441 | 0.347 | 0.236 |
| always the most common label | 0.152 | 0.283 | 0.762 | 0.635 | 0.618 | 0.502 | 0.549 | 0.373 | 0.262 |

Jev is ahead on average, and almost all of its lead comes from the knowledge tasks (0.869 against 0.742
for Gemma 4 12B); on the other 22 tasks the two score 0.763 and 0.761. Gaps of less than about two
points are not reliable over 25 tasks. Before any temperature, Jev's probabilities are already close to
calibrated (raw ECE 0.085), while every other model above 0.6 accuracy is overconfident (raw ECE
0.12–0.23). hateday_tr, checkthat_tr and trclaim19_tr are imbalanced, so accuracy in safety and
fact-checking mostly shows how often a model flags; in fact-checking every model is below always
answering the most common label. By macro-F1, Jev leads safety (0.767, Gemma 4 12B 0.721, Metask-Jev-4B
0.701) and Gemma 4 12B leads fact-checking (0.532, Qwen3.5 9B 0.499, Metask-Jev-4B 0.495, Jev 0.486).
kumru-2b and mecellem-qwen3-4b score below always answering the most common label. Every task and
question, with raw NLL and Brier, is in `results/scores.csv`.

## Item

```json
{
  "id": "nli_tr/test/00003",
  "task": "nli_tr",
  "split": "test",
  "state": {"öncül": "...", "hipotez": "..."},
  "questions": {
    "relation": {
      "type": "choice",
      "instructions": "Öncül doğru kabul edilirse hipotez hakkında ne söylenebilir?",
      "criteria": {"entailment": "...", "neutral": "...", "contradiction": "..."}
    }
  },
  "gold": {"relation": "neutral"},
  "soft_gold": {"relation": {"entailment": 0.2, "neutral": 0.6, "contradiction": 0.2}}
}
```

Gold is the option name for Choice, `true`/`false` for Noul and the 0-based level for Score.
`soft_gold` holds annotators' votes where they are known: MultiNLI's five annotators in nli_tr, the
share of three annotators who found a tweet worth checking in trclaim19_tr, the vote shares in
biltweetnews_tr, the three raters of rtp_lx_tr and the annotator shares that jev-bench keeps for
ChaosNLI, Civil Comments, GoEmotions and Measuring Hate Speech. In lavoir_zeroshot_tr it holds the
exact probabilities the dialogues were generated from. An item can ask several questions: seahorse_tr asks six about each summary (only
the first when the raters could not understand it), checkthat_tr up to three about each tweet and
touche_tr up to two about each speech.
Some tasks add `meta`: cultural sensitivity in Global-MMLU, workflow and inversion in jev_noul, the
raw 0–5 score in STSb.

## Tasks

| Task | Area | Type | Options | Dev | Test | Source | Licence |
|---|---|---|---|---|---|---|---|
| massive_intent_tr | routing | Choice | 60 | 200 | 1000 | [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) tr-TR, via the `mteb/amazon_massive_intent` parquet copy | CC BY 4.0 |
| xsid_tr | routing | Choice | 16 | 200 | 500 | [xSID](https://github.com/mainlp/xsid) 0.6, Turkish | CC BY-SA 4.0 |
| news_topic_tr | routing | Choice | 7 | 200 | 1000 | [TTC-4900](https://huggingface.co/datasets/savasy/ttc4900) news categories | unknown |
| buyuksinema_tr | opinion | Score | 10 | 200 | 1000 | [BuyukSinema](https://huggingface.co/datasets/turkish-nlp-suite/BuyukSinema) | CC BY-SA 4.0 |
| musteri_yorumlari_tr | opinion | Score | 5 | 200 | 1000 | [MusteriYorumlari](https://huggingface.co/datasets/turkish-nlp-suite/MusteriYorumlari) | CC BY-SA 4.0 |
| biltweetnews_tr | opinion | Choice | 4 | 95 | 863 | [BilTweetNews](https://huggingface.co/datasets/ctoraman/BilTweetNews-sentiment-analysis) | CC BY-NC-SA 4.0 |
| tweet_emotion_tr | opinion | Choice | 5 | 200 | 1000 | [Turkish tweet emotion](https://huggingface.co/datasets/anilguven/turkish_tweet_emotion_dataset) | unknown |
| offenseval_tr | safety | Noul | | 200 | 1000 | [OffensEval-2020 TR](https://coltekin.github.io/offensive-turkish/) | CC BY 2.0 |
| sms_spam_tr | safety | Noul | | 200 | 1000 | [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) | unknown |
| hateday_tr | safety | Choice | 3 | 200 | 10000 | [HateDay](https://huggingface.co/datasets/manueltonneau/hateday), Turkish random sample | gated, not stated |
| xfact_teyit_tr | fact_checking | Choice | 4 | 61 | 552 | X-FACT out-of-domain, teyit.org | MIT |
| checkthat_tr | fact_checking | Noul ×3 | | 200 | 1000 | [CheckThat! 2022](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab) Turkish, subtasks 1A–1C | not stated |
| trclaim19_tr | fact_checking | Noul | | 200 | 1000 | [TrClaim-19](https://huggingface.co/datasets/mcemilg/TrClaim19) | not stated |
| seahorse_tr | judging | Noul ×6 | | 200 | 1000 | [SEAHORSE](https://huggingface.co/datasets/tasksource/seahorse_summarization_evaluation), Turkish raters | CC BY 4.0 |
| webfaq_tr | judging | Noul | | 200 | 1000 | [WebFAQ](https://huggingface.co/datasets/mteb/WebFAQRetrieval) Turkish test queries | CC BY 4.0 |
| jev_noul_tr | workflow | Noul | | 200 | 1000 | [turkish_jev_noul](https://huggingface.co/datasets/ahmetege/turkish_jev_noul) | CC BY-SA 4.0 |
| jev_noul_unseen_tr | workflow | Noul | | 100 | 900 | turkish_jev_noul, unseen-task split | CC BY-SA 4.0 |
| touche_tr | politics | Noul + Choice | 2 | 200 | 1000 | [Touché 2024 ideology and power](https://zenodo.org/records/10450641), ParlaMint-TR speeches | CC BY 4.0 |
| nli_tr | language | Choice | 3 | 200 | 1000 | [NLI-TR](https://github.com/boun-tabi/NLI-TR) MultiNLI-TR | MultiNLI terms |
| stsb_tr | language | Score | 6 | 200 | 1000 | [STSb-TR](https://github.com/verimsu/STSb-TR) | unspecified |
| belebele_tr | language | Choice | 4 | 90 | 810 | [Belebele](https://huggingface.co/datasets/facebook/belebele) tur_Latn | CC BY-SA 4.0 |
| xcopa_tr | language | Choice | 2 | 100 | 500 | [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa) tr | CC BY 4.0 |
| turblimp_tr | language | Choice | 2 | 200 | 1000 | [TurBLiMP](https://huggingface.co/datasets/juletxara/turblimp) minimal pairs | CC BY 4.0 |
| include_tr | knowledge | Choice | 4 | 12 | 419 | [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44), Turkish exams, without the questions it repeats from TurkishMMLU | Apache-2.0 |
| global_mmlu_tr | knowledge | Choice | 4 | 200 | 1000 | [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) tr, 500 culturally sensitive + 500 agnostic | Apache-2.0 |
| turkishmmlu_tr | knowledge | Choice | 5 | 90 | 810 | [TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU), public test | not stated |

Test is a random sample of at most 1,000 items from the source's test split (seed 13). Dev holds up
to 200 items from the source's dev split and is meant for fitting a temperature per model. Sources
without a dev split give 10% of their test items to dev. A few questions appear in both a source's
dev and test split.

In X-FACT, evidence snippets from or about fact-checking sites, and snippets elsewhere that state a
verdict ("...iddiası doğru değil"), are dropped because they give the verdict away. include_tr leaves
out the 137 TurkishMMLU questions INCLUDE repeats with one option dropped, often the right one, and an
answer key that disagrees with TurkishMMLU's; turkishmmlu_tr asks them properly.

In the tweet tasks, user names are replaced with `@USER`. checkthat_tr takes its test tweets from
CheckThat!'s test and dev-test sets. In webfaq_tr, half of the questions come with their own answer
and half with the answer of the most similar other question, leaving out near-copies of the question.
SEAHORSE stores its summaries JSON-escaped, so they are decoded first; a few article and model pairs
have two different summaries, and only the first one's ratings are used. aym_outcome_tr keeps the
first 8,000 characters of each application; a true label means the Court found at least one violation,
which we checked against the Court's decision database.

hateday_tr tests on 10,000 tweets of HateDay's random sample, so hate (about 0.6%) and offence
(about 4%) appear at their real rates; the engagement-weighted sample goes to rest. touche_tr keeps
the first 8,000 characters of each speech and asks whether the speaker's party governs and whether
it is on the left or the right, as the task defines them. turblimp_tr drops the few pairs whose two
sentences are identical and alternates which sentence comes first. turkishmmlu_difficulty_tr uses
TurkishMMLU's own easy, medium and hard labels, which come from the share of students who answered
correctly.

## Secondary tasks

These are translated, synthetic or very small, or no model reads their label from the state, so they
are reported apart from the tasks above and left out of the area averages. They are listed in
`SECONDARY` in `s1bench/tasks/__init__.py`.

| Task | Type | Dev | Test | Source | Why secondary | Licence |
|---|---|---|---|---|---|---|
| jevbench_<config>_tr (22 tasks) | all three | 68–200 | 619–1000 | [jev-bench-tr](https://huggingface.co/datasets/hayriyigit/jev-bench-tr) | machine-translated; compares with Jev's published English scores | upstream licences |
| bev_tr | all three | 200 | 1000 | [BEV Decision Mix, Turkish](https://huggingface.co/datasets/hayriyigit/bev-decision-150K-tr) | machine-translated | unknown |
| lavoir_zeroshot_tr | Choice | 200 | 1000 | [Lavoir dialogues](https://huggingface.co/datasets/moganai/lavoir-dialogues-tr), zero-shot workflows | synthetic | CC BY 4.0 |
| rtp_lx_tr | Score | 104 | 945 | [RTP-LX](https://huggingface.co/datasets/ToxicityPrompts/RTP-LX), Turkish prompts | prompts transcreated from English | ODC-By |
| injection_tr | Noul | 97 | 113 | [turkish-prompt-injection-1k](https://huggingface.co/datasets/3nesdeniz/turkish-prompt-injection-1k) | synthetic | CC BY 4.0 |
| over_refusal_tr | Noul | 24 | 216 | [turkish-over-refusal-set](https://huggingface.co/datasets/fevziegeyurtsevenler/turkish-over-refusal-set) | small | Apache-2.0 |
| xl_safety_tr | Noul | 15 | 135 | [XL-SafetyBench](https://huggingface.co/datasets/AIM-Intelligence/XL-SafetyBench), Turkey | every request is harmful, so it measures recall only | CC BY 4.0 |
| ragtruth_tr | Noul | 200 | 1000 | [RAGTruth-TR](https://huggingface.co/datasets/newmindai/RAGTruth-TR), test split | machine-translated | MIT |
| rewardbench_tr | Choice | 200 | 1000 | [M-RewardBench](https://huggingface.co/datasets/CohereLabsCommunity/multilingual-reward-bench) tur_Latn | machine-translated | ODC-By |
| wmt_da_tr | Score | 200 | 1000 | [WMT human ratings](https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation), English to Turkish | English source texts | Apache-2.0 |
| tool_choice_tr | Choice | 200 | 1000 | [Turkish mobile function calling](https://huggingface.co/datasets/BTX24/turkish-mobile-function-calling-dataset) | probably generated | Apache-2.0 |
| irony_tr | Noul | 60 | 540 | [IronyTR](https://huggingface.co/datasets/mcemilg/IronyTR) | small | not stated |
| neural_news_tr | Noul | 200 | 1000 | [Neural news benchmark](https://huggingface.co/datasets/tum-nlp/neural-news-benchmark), Turkish | half of the articles are generated | OpenRAIL++ |
| tus21_tr | Choice | 46 | 419 | [TUS 2021](https://huggingface.co/datasets/zypchn/TUS21-exams) medical exam | small | not stated |
| global_piqa_tr | Choice | 10 | 90 | [Global PIQA](https://huggingface.co/datasets/mrlbenchmarks/global-piqa-nonparallel) tur_latn | 100 items | CC BY-SA 4.0 |
| xfact_tr | Choice | 111 | 169 | [X-FACT](https://github.com/utahnlp/x-fact), Doğruluk Payı | most claims turn on numbers or dates the state lacks; every model is near always answering the most common verdict | MIT |
| mide22_tr | Choice | 200 | 1000 | [MiDe22](https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection) | labels are relative to the event a tweet was collected for, so debunking tweets count as true; every model is near the majority | MIT |
| turkishmmlu_difficulty_tr | Score | 90 | 810 | [TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU), share of students who answered correctly | the label describes one platform's students, not the question; every model is below the majority | not stated |
| aym_outcome_tr | Noul | 200 | 665 | [Constitutional Court individual applications](https://huggingface.co/datasets/icgcihan/Turkish_Constutional_Court_Decisions) | the outcome cannot be read from the application; every model is at chance | CC BY 4.0 |

In neural_news_tr the English prompt that was used to generate an article is turned back into its
headline and lead, so the prompt itself does not give the answer away; what the models generated is
kept as it is.

## Not included yet

The [physics short-answer grades](https://www.kaggle.com/datasets/elifince/ml-for-openended-physics-questions-in-turkish)
hold the students' answers but not the questions they answered, so a grader cannot be judged fairly
on them.

[Turkish tool-call traces](https://huggingface.co/datasets/aliarda/jev_turkish_tool_call_traces) is
gated and its author approves access by hand. If its labels came from Jev, it would not be a fair
test for Jev.

## Seen data

Laya, Verdict, Jev-Style-0.8B and Metask-Jev-4B were trained on MASSIVE, and some on XNLI or
MultiNLI. Belebele, XCOPA and MMLU are old and widely used, so large LLMs have probably seen them.
Models we train never see a text from `data/`, rest included: `s1bench.fingerprints` and the training
data's pull remove copies. They do train on other data from some of the same sources: TrGLUE's train
split (jev_noul_tr's dev and test come from TrGLUE's validation and test sets) and its STS-B, Lavoir's
seen workflows, WebFAQ questions from other sites, FACTurk fact-checks with the benchmark's claims
removed, the Kemik news collection that TTC-4900 comes from, and reviews from the sites behind
MusteriYorumlari and BuyukSinema. Their scores on jev_noul_tr, stsb_tr, lavoir_zeroshot_tr, webfaq_tr,
xfact_teyit_tr, news_topic_tr, musteri_yorumlari_tr and buyuksinema_tr are in-distribution.
turkish_jev_noul's train split holds every belebele_tr item with its answer.

`data/` is not committed because several sources restrict redistribution; anyone can rebuild it.
