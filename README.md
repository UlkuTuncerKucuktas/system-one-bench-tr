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
probabilities and the votes. It also prints accuracy averaged over the tasks of each area.

Models are listed in `s1bench/models.py`:

- Ordinary LLMs see the state, the question and lettered options, and we read the probability of
  each letter as the next token in one forward pass, with thinking turned off. Choice and Noul
  questions are asked twice with the options in reverse order and the two answers are averaged.
- Metask-Jev-4B uses its own prompt format and handles at most 26 options, so it skips MASSIVE.
- Laya uses its multilingual checkpoint.
- Jev needs your key in `TYPESAFE_API_KEY`.

## Keeping training data out of the benchmark

```bash
python -m s1bench.fingerprints
```

This writes `data/fingerprints.txt`: a hash of every 8-word window of every dev and test text, and a
hash of every whole text in `rest.jsonl`. A training text touches the benchmark when
`s1bench.fingerprints.fingerprints(text)` shares a hash with that file. Models we train only see text
that shares none.

## Results

| Model | Accuracy (12 tasks) | Macro-F1 (12 tasks) | ECE (12 tasks) | MASSIVE accuracy | Score MAE |
|---|---|---|---|---|---|
| `jev` | 0.725 | 0.708 | 0.084 | 0.788 | 0.86 |
| `gemma-4-12b` | 0.709 | 0.687 | 0.059 | 0.777 | 0.82 |
| `qwen3.5-35b-a3b` | 0.701 | 0.665 | 0.056 | 0.771 | 0.89 |
| `trendyol-asure-12b` | 0.675 | 0.653 | 0.060 | 0.685 | 0.89 |
| `qwen3.5-9b` | 0.662 | 0.626 | 0.058 | 0.691 | 1.03 |
| `gemma-4-e4b` | 0.655 | 0.635 | 0.063 | 0.720 | 0.97 |
| `turkish-gemma-9b` | 0.644 | 0.616 | 0.073 | 0.652 | 1.12 |
| `metask-jev-4b` | 0.630 | 0.593 | 0.079 | – | 0.98 |
| `qwen3.5-4b` | 0.616 | 0.576 | 0.058 | 0.570 | 1.35 |
| `eurollm-9b` | 0.583 | 0.546 | 0.067 | 0.565 | 1.45 |
| `kizagan-e4b` | 0.562 | 0.535 | 0.048 | 0.514 | 1.46 |
| `laya-multilingual` | 0.413 | 0.377 | 0.052 | 0.347 | 1.67 |
| `mecellem-qwen3-4b` | 0.382 | 0.304 | 0.056 | 0.006 | 1.67 |
| `kumru-2b` | 0.332 | 0.263 | 0.037 | 0.003 | 1.69 |

These numbers cover the first 14 tasks. The nine tasks added later have not been run yet.

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
share of three annotators who found a tweet worth checking in trclaim19_tr, and the vote shares in
biltweetnews_tr. An item can ask several questions: seahorse_tr asks six about each summary (only
the first when the raters could not understand it) and checkthat_tr up to three about each tweet.
Some tasks add `meta`: cultural sensitivity in Global-MMLU, workflow and inversion in jev_noul, the
raw 0–5 score in STSb.

## Tasks

| Task | Area | Type | Options | Dev | Test | Source | Licence |
|---|---|---|---|---|---|---|---|
| massive_intent_tr | intent | Choice | 60 | 200 | 1000 | [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) tr-TR, via the `mteb/amazon_massive_intent` parquet copy | CC BY 4.0 |
| xsid_tr | intent | Choice | 16 | 200 | 500 | [xSID](https://github.com/mainlp/xsid) 0.6, Turkish | CC BY-SA 4.0 |
| buyuksinema_tr | opinion | Score | 10 | 200 | 1000 | [BuyukSinema](https://huggingface.co/datasets/turkish-nlp-suite/BuyukSinema) | CC BY-SA 4.0 |
| musteri_yorumlari_tr | opinion | Score | 5 | 200 | 1000 | [MusteriYorumlari](https://huggingface.co/datasets/turkish-nlp-suite/MusteriYorumlari) | CC BY-SA 4.0 |
| biltweetnews_tr | opinion | Choice | 4 | 95 | 863 | [BilTweetNews](https://huggingface.co/datasets/ctoraman/BilTweetNews-sentiment-analysis) | CC BY-NC-SA 4.0 |
| offenseval_tr | safety | Noul | | 200 | 1000 | [OffensEval-2020 TR](https://coltekin.github.io/offensive-turkish/) | CC BY 2.0 |
| sms_spam_tr | safety | Noul | | 200 | 1000 | [Turkish SMS Collection](https://www.kaggle.com/datasets/onurkarasoy/turkish-sms-collection) | unknown |
| xfact_tr | fact_checking | Choice | 5 | 111 | 169 | [X-FACT](https://github.com/utahnlp/x-fact), Doğruluk Payı | MIT |
| xfact_teyit_tr | fact_checking | Choice | 4 | 61 | 552 | X-FACT out-of-domain, teyit.org | MIT |
| mide22_tr | fact_checking | Choice | 3 | 200 | 1000 | [MiDe22](https://huggingface.co/datasets/ogozcelik/turkish-fake-news-detection) | MIT |
| checkthat_tr | fact_checking | Noul ×3 | | 200 | 1000 | [CheckThat! 2022](https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab) Turkish, subtasks 1A–1C | not stated |
| trclaim19_tr | fact_checking | Noul | | 200 | 1000 | [TrClaim-19](https://huggingface.co/datasets/mcemilg/TrClaim19) | not stated |
| seahorse_tr | judging | Noul ×6 | | 200 | 1000 | [SEAHORSE](https://huggingface.co/datasets/tasksource/seahorse_summarization_evaluation), Turkish raters | CC BY 4.0 |
| webfaq_tr | judging | Noul | | 200 | 1000 | [WebFAQ](https://huggingface.co/datasets/mteb/WebFAQRetrieval) Turkish test queries | CC BY 4.0 |
| jev_noul_tr | workflow | Noul | | 200 | 1000 | [turkish_jev_noul](https://huggingface.co/datasets/ahmetege/turkish_jev_noul) | CC BY-SA 4.0 |
| jev_noul_unseen_tr | workflow | Noul | | 100 | 900 | turkish_jev_noul, unseen-task split | CC BY-SA 4.0 |
| aym_outcome_tr | legal | Noul | | 200 | 665 | [Constitutional Court individual applications](https://huggingface.co/datasets/icgcihan/Turkish_Constutional_Court_Decisions) | CC BY 4.0 |
| nli_tr | language | Choice | 3 | 200 | 1000 | [NLI-TR](https://github.com/boun-tabi/NLI-TR) MultiNLI-TR | MultiNLI terms |
| stsb_tr | language | Score | 6 | 200 | 1000 | [STSb-TR](https://github.com/verimsu/STSb-TR) | unspecified |
| belebele_tr | language | Choice | 4 | 90 | 810 | [Belebele](https://huggingface.co/datasets/facebook/belebele) tur_Latn | CC BY-SA 4.0 |
| xcopa_tr | language | Choice | 2 | 100 | 500 | [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa) tr | CC BY 4.0 |
| include_tr | knowledge | Choice | 4 | 20 | 548 | [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44), Turkish exams | Apache-2.0 |
| global_mmlu_tr | knowledge | Choice | 4 | 200 | 1000 | [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) tr, 500 culturally sensitive + 500 agnostic | Apache-2.0 |

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

## Not included yet

[HateDay](https://huggingface.co/datasets/manueltonneau/hateday) is gated; its columns can only be
read after accepting its terms, so its task comes after that.

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
