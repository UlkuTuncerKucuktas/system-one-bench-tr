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
probabilities and the votes. A question without dev items (a few in bev_tr) gets the temperature
fitted on the whole task. It also prints accuracy averaged over the tasks of each area; the
secondary tasks are left out of those averages.

Models are listed in `s1bench/models.py`:

- Ordinary LLMs see the state, the question and lettered options, and we read the probability of
  each letter as the next token in one forward pass, with thinking turned off. Choice and Noul
  questions are asked twice with the options in reverse order and the two answers are averaged.
  The letters are A–Z, a–z and 0–9, so a question with more than 62 options is skipped (banking77,
  clinc150 and ledgar in jev-bench-tr).
- Metask-Jev-4B uses its own prompt format and handles at most 26 options, so besides those it skips
  MASSIVE and jev-bench-tr's MASSIVE and GoEmotions. It reads prompts of up to 32,768 tokens, since
  the other models read the whole state too.
- Laya uses its multilingual checkpoint.
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

Accuracy, macro-F1 and ECE average the 29 area tasks that every model answered (all but MASSIVE, which
Metask skips); each task counts once, with its questions averaged. Score MAE averages the four Score
tasks. The last two columns average the 14 secondary tasks outside jev-bench-tr and the 17
jev-bench-tr tasks that every LLM and Metask answered. Jev was run on the area tasks only.

| Model | Accuracy | Macro-F1 | ECE | MASSIVE accuracy | Score MAE | Secondary accuracy | jev-bench-tr accuracy |
|---|---|---|---|---|---|---|---|
| `jev` | 0.720 | 0.688 | 0.068 | 0.788 | 0.79 | – | – |
| `gemma-4-12b` | 0.704 | 0.668 | 0.053 | 0.777 | 0.77 | 0.759 | 0.637 |
| `qwen3.5-35b-a3b` | 0.689 | 0.647 | 0.050 | 0.771 | 0.81 | 0.754 | 0.655 |
| `metask-jev-4b` | 0.655 | 0.610 | 0.066 | – | 0.88 | 0.661 | 0.654 |
| `qwen3.5-9b` | 0.655 | 0.608 | 0.054 | 0.691 | 0.92 | 0.704 | 0.626 |
| `trendyol-asure-12b` | 0.654 | 0.615 | 0.066 | 0.685 | 0.82 | 0.677 | 0.579 |
| `gemma-4-e4b` | 0.642 | 0.605 | 0.056 | 0.720 | 0.88 | 0.660 | 0.579 |
| `turkish-gemma-9b` | 0.615 | 0.577 | 0.079 | 0.652 | 0.99 | 0.656 | 0.547 |
| `qwen3.5-4b` | 0.613 | 0.564 | 0.054 | 0.570 | 1.16 | 0.642 | 0.623 |
| `eurollm-9b` | 0.578 | 0.509 | 0.073 | 0.565 | 1.23 | 0.590 | 0.527 |
| `kizagan-e4b` | 0.565 | 0.523 | 0.056 | 0.514 | 1.24 | 0.574 | 0.496 |
| `laya-multilingual` | 0.441 | 0.393 | 0.061 | 0.347 | 1.40 | 0.448 | 0.417 |
| `kumru-2b` | 0.359 | 0.276 | 0.057 | 0.003 | 1.43 | 0.405 | 0.365 |
| `mecellem-qwen3-4b` | 0.355 | 0.277 | 0.064 | 0.006 | 1.40 | 0.375 | 0.375 |

Accuracy by area, over all the tasks of each area:

| Model | routing | opinion | safety | fact_checking | judging | workflow | legal | politics | language | knowledge |
|---|---|---|---|---|---|---|---|---|---|---|
| `jev` | 0.874 | 0.624 | 0.906 | 0.470 | 0.654 | 0.873 | 0.541 | 0.866 | 0.789 | 0.810 |
| `gemma-4-12b` | 0.862 | 0.614 | 0.865 | 0.508 | 0.658 | 0.873 | 0.493 | 0.852 | 0.778 | 0.687 |
| `qwen3.5-35b-a3b` | 0.858 | 0.598 | 0.822 | 0.459 | 0.624 | 0.860 | 0.564 | 0.858 | 0.759 | 0.735 |
| `metask-jev-4b` | – | 0.536 | 0.914 | 0.507 | 0.622 | 0.789 | 0.432 | 0.740 | 0.681 | 0.583 |
| `qwen3.5-9b` | 0.811 | 0.545 | 0.806 | 0.480 | 0.633 | 0.842 | 0.456 | 0.767 | 0.701 | 0.652 |
| `trendyol-asure-12b` | 0.831 | 0.621 | 0.716 | 0.384 | 0.639 | 0.871 | 0.480 | 0.829 | 0.753 | 0.623 |
| `gemma-4-e4b` | 0.832 | 0.591 | 0.817 | 0.406 | 0.592 | 0.861 | 0.481 | 0.800 | 0.691 | 0.583 |
| `turkish-gemma-9b` | 0.813 | 0.584 | 0.557 | 0.373 | 0.610 | 0.860 | 0.454 | 0.863 | 0.688 | 0.621 |
| `qwen3.5-4b` | 0.759 | 0.499 | 0.677 | 0.436 | 0.635 | 0.829 | 0.496 | 0.639 | 0.679 | 0.592 |
| `eurollm-9b` | 0.750 | 0.520 | 0.712 | 0.385 | 0.547 | 0.842 | 0.531 | 0.734 | 0.601 | 0.448 |
| `kizagan-e4b` | 0.739 | 0.463 | 0.712 | 0.350 | 0.594 | 0.762 | 0.471 | 0.655 | 0.639 | 0.438 |
| `laya-multilingual` | 0.569 | 0.303 | 0.539 | 0.366 | 0.502 | 0.675 | 0.471 | 0.495 | 0.446 | 0.243 |
| `kumru-2b` | 0.065 | 0.158 | 0.643 | 0.373 | 0.453 | 0.515 | 0.528 | 0.439 | 0.347 | 0.234 |
| `mecellem-qwen3-4b` | 0.088 | 0.160 | 0.385 | 0.349 | 0.478 | 0.550 | 0.514 | 0.487 | 0.420 | 0.292 |

Jev is ahead on average, mostly on the knowledge tasks (TurkishMMLU 0.90 and Global-MMLU 0.85, against
0.71 and 0.70 for Gemma 4 12B). Metask-Jev-4B is the best model on safety. Fact-checking is the hardest
area for every model, and kumru-2b and mecellem-qwen3-4b score below always answering the most common
label. The numbers for every task and question are in `results/scores.csv`.

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
| xfact_tr | fact_checking | Choice | 5 | 111 | 169 | [X-FACT](https://github.com/utahnlp/x-fact), Doğruluk Payı | MIT |
| xfact_teyit_tr | fact_checking | Choice | 4 | 61 | 552 | X-FACT out-of-domain, teyit.org | MIT |
| mide22_tr | fact_checking | Choice | 3 | 200 | 1000 | [MiDe22](https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection) | MIT |
| checkthat_tr | fact_checking | Noul ×3 | | 200 | 1000 | [CheckThat! 2022](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab) Turkish, subtasks 1A–1C | not stated |
| trclaim19_tr | fact_checking | Noul | | 200 | 1000 | [TrClaim-19](https://huggingface.co/datasets/mcemilg/TrClaim19) | not stated |
| seahorse_tr | judging | Noul ×6 | | 200 | 1000 | [SEAHORSE](https://huggingface.co/datasets/tasksource/seahorse_summarization_evaluation), Turkish raters | CC BY 4.0 |
| webfaq_tr | judging | Noul | | 200 | 1000 | [WebFAQ](https://huggingface.co/datasets/mteb/WebFAQRetrieval) Turkish test queries | CC BY 4.0 |
| turkishmmlu_difficulty_tr | judging | Score | 3 | 90 | 810 | [TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU), share of students who answered correctly | not stated |
| jev_noul_tr | workflow | Noul | | 200 | 1000 | [turkish_jev_noul](https://huggingface.co/datasets/ahmetege/turkish_jev_noul) | CC BY-SA 4.0 |
| jev_noul_unseen_tr | workflow | Noul | | 100 | 900 | turkish_jev_noul, unseen-task split | CC BY-SA 4.0 |
| aym_outcome_tr | legal | Noul | | 200 | 665 | [Constitutional Court individual applications](https://huggingface.co/datasets/icgcihan/Turkish_Constutional_Court_Decisions) | CC BY 4.0 |
| touche_tr | politics | Noul + Choice | 2 | 200 | 1000 | [Touché 2024 ideology and power](https://zenodo.org/records/10450641), ParlaMint-TR speeches | CC BY 4.0 |
| nli_tr | language | Choice | 3 | 200 | 1000 | [NLI-TR](https://github.com/boun-tabi/NLI-TR) MultiNLI-TR | MultiNLI terms |
| stsb_tr | language | Score | 6 | 200 | 1000 | [STSb-TR](https://github.com/verimsu/STSb-TR) | unspecified |
| belebele_tr | language | Choice | 4 | 90 | 810 | [Belebele](https://huggingface.co/datasets/facebook/belebele) tur_Latn | CC BY-SA 4.0 |
| xcopa_tr | language | Choice | 2 | 100 | 500 | [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa) tr | CC BY 4.0 |
| turblimp_tr | language | Choice | 2 | 200 | 1000 | [TurBLiMP](https://huggingface.co/datasets/juletxara/turblimp) minimal pairs | CC BY 4.0 |
| include_tr | knowledge | Choice | 4 | 20 | 548 | [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44), Turkish exams | Apache-2.0 |
| global_mmlu_tr | knowledge | Choice | 4 | 200 | 1000 | [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) tr, 500 culturally sensitive + 500 agnostic | Apache-2.0 |
| turkishmmlu_tr | knowledge | Choice | 5 | 90 | 810 | [TurkishMMLU](https://huggingface.co/datasets/AYueksel/TurkishMMLU), public test | not stated |

Test is a random sample of at most 1,000 items from the source's test split (seed 13). Dev holds up
to 200 items from the source's dev split and is meant for fitting a temperature per model. Sources
without a dev split give 10% of their test items to dev. A few questions appear in both a source's
dev and test split.

In X-FACT, evidence snippets from or about fact-checking sites are dropped because they give the
verdict away.

In the tweet tasks, user names are replaced with `@USER`. checkthat_tr takes its test tweets from
CheckThat!'s test and dev-test sets. In webfaq_tr, half of the questions come with their own answer
and half with the answer of the most similar other question, leaving out near-copies of the
question. SEAHORSE stores its summaries JSON-escaped, so they are decoded first. aym_outcome_tr keeps
the first 8,000 characters of each application; a true label means the Court found at least one
violation, which we checked against the Court's decision database.

hateday_tr tests on 10,000 tweets of HateDay's random sample, so hate (about 0.6%) and offence
(about 4%) appear at their real rates; the engagement-weighted sample goes to rest. touche_tr keeps
the first 8,000 characters of each speech and asks whether the speaker's party governs and whether
it is on the left or the right, as the task defines them. turblimp_tr drops the few pairs whose two
sentences are identical and alternates which sentence comes first. turkishmmlu_difficulty_tr uses
TurkishMMLU's own easy, medium and hard labels, which come from the share of students who answered
correctly.

## Secondary tasks

These are translated, synthetic or very small, so they are reported apart from the tasks above and
left out of the area averages. They are listed in `SECONDARY` in `s1bench/tasks/__init__.py`.

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
Models we train are kept away from everything in `data/`, rest included, with `s1bench.fingerprints`.

`data/` is not committed because several sources restrict redistribution; anyone can rebuild it.
