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

This writes `data/<task>/dev.jsonl` and `data/<task>/test.jsonl`. To rebuild some tasks only:
`python -m s1bench.build nli_tr xcopa_tr`.

## Evaluate

```bash
python -m s1bench.evaluate qwen3.5-4b
python -m s1bench.score
```

`evaluate` runs one model on every task (or on the tasks you list after the model name) and writes
its probabilities to `results/<model>/<task>/`. `score` fits one temperature per model and task on
dev, then writes accuracy, Brier score, calibration error (ECE) and, for Score tasks, the mean
absolute error of the expected level to `results/scores.csv`.

Models are listed in `s1bench/models.py`:

- Ordinary LLMs see the state, the question and lettered options, and we read the probability of
  each letter as the next token in one forward pass, with thinking turned off. Choice and Noul
  questions are asked twice with the options in reverse order and the two answers are averaged.
- Metask-Jev-4B uses its own prompt format and handles at most 26 options, so it skips MASSIVE.
- Laya uses its multilingual checkpoint.
- Jev needs your key in `TYPESAFE_API_KEY`.

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
`soft_gold` (nli_tr only) holds the votes of MultiNLI's five annotators. Some tasks add `meta`:
cultural sensitivity in Global-MMLU, workflow and inversion in jev_noul, the raw 0–5 score in STSb.

## Tasks

| Task | Type | Options | Dev | Test | Source | Licence |
|---|---|---|---|---|---|---|
| massive_intent_tr | Choice | 60 | 200 | 1000 | [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) tr-TR, via the `mteb/amazon_massive_intent` parquet copy | CC BY 4.0 |
| belebele_tr | Choice | 4 | 90 | 810 | [Belebele](https://huggingface.co/datasets/facebook/belebele) tur_Latn | CC BY-SA 4.0 |
| xcopa_tr | Choice | 2 | 100 | 500 | [XCOPA](https://huggingface.co/datasets/cambridgeltl/xcopa) tr | CC BY 4.0 |
| include_tr | Choice | 4 | 20 | 548 | [INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44), Turkish exams | Apache-2.0 |
| global_mmlu_tr | Choice | 4 | 200 | 1000 | [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) tr, 500 culturally sensitive + 500 agnostic | Apache-2.0 |
| nli_tr | Choice | 3 | 200 | 1000 | [NLI-TR](https://github.com/boun-tabi/NLI-TR) MultiNLI-TR | MultiNLI terms |
| xfact_tr | Choice | 5 | 111 | 169 | [X-FACT](https://github.com/utahnlp/x-fact), Doğruluk Payı | MIT |
| xfact_teyit_tr | Choice | 4 | 61 | 552 | X-FACT out-of-domain, teyit.org | MIT |
| offenseval_tr | Noul | | 200 | 1000 | [OffensEval-2020 TR](https://coltekin.github.io/offensive-turkish/) | CC BY 2.0 |
| jev_noul_tr | Noul | | 200 | 1000 | [turkish_jev_noul](https://huggingface.co/datasets/ahmetege/turkish_jev_noul) | CC BY-SA 4.0 |
| jev_noul_unseen_tr | Noul | | 100 | 900 | turkish_jev_noul, unseen-task split | CC BY-SA 4.0 |
| buyuksinema_tr | Score | 10 | 200 | 1000 | [BuyukSinema](https://huggingface.co/datasets/turkish-nlp-suite/BuyukSinema) | CC BY-SA 4.0 |
| musteri_yorumlari_tr | Score | 5 | 200 | 1000 | [MusteriYorumlari](https://huggingface.co/datasets/turkish-nlp-suite/MusteriYorumlari) | CC BY-SA 4.0 |
| stsb_tr | Score | 6 | 200 | 1000 | [STSb-TR](https://github.com/verimsu/STSb-TR) | unspecified |

Test is a random sample of at most 1,000 items from the source's test split (seed 13). Dev holds up
to 200 items from the source's dev split and is meant for fitting a temperature per model. Sources
without a dev split give 10% of their test items to dev. A few questions appear in both a source's
dev and test split.

In X-FACT, evidence snippets from or about fact-checking sites are dropped because they give the
verdict away.

## Not included yet

[Turkish tool-call traces](https://huggingface.co/datasets/aliarda/jev_turkish_tool_call_traces) is
gated and its author approves access by hand. Once you have access, log in with
`huggingface-cli login` and add a task for it.

## Seen data

Laya, Verdict, Jev-Style-0.8B and Metask-Jev-4B were trained on MASSIVE, and some on XNLI or
MultiNLI. Belebele, XCOPA and MMLU are old and widely used, so large LLMs have probably seen them.

`data/` is not committed because several sources restrict redistribution; anyone can rebuild it.
