# system-one-bench-tr

A Turkish benchmark of 66 classification-style tasks in the System One request format. A model reads a text and a
question with named answer options, and its answer is read from the probabilities it gives the options' letters in one
forward pass.

## Background

Jev, TypeSafe's System One model, answers requests of this form: a text (a message, a document or a record with named
fields) and one or more questions, each of one kind:

- **Choice**: one of several named options, each with a criterion that says when it applies.
- **Noul**: yes or no, with a criterion for each.
- **Score**: a level on an ordered scale, each level described.

General language models can be asked in the same way. We built this benchmark to compare Jev, Microsoft-Decision-1, open
models and our own fine-tuned models on the same requests. Because the answers are probabilities, the benchmark reports
calibration as well as accuracy. HakemBench [1], released at about the same time, is another Turkish benchmark in this
format.

## Tasks

66 tasks built from public Turkish datasets, each turned into System One requests with a dev and a test split. 26 tasks
are grouped into nine areas; the 25 main tasks are these without MASSIVE, which one of the compared models cannot
answer. The other 40 tasks are secondary, among them jev-bench-tr, a Turkish version of 22 jev-bench tasks.

| Area | Tasks |
|---|---|
| Routing | intent (MASSIVE, xSID), news topic (TTC-4900) |
| Opinion | movie and product reviews (BüyükSinema, Müşteri Yorumları), news tweets (BilTweetNews), tweet emotion |
| Safety | offensive language (OffensEval), SMS spam, hate speech (HateDay) |
| Fact-checking | claim verdicts (X-FACT, teyit.org), check-worthy claims (CheckThat, TrClaim-19) |
| Judging | summary quality (SEAHORSE), answers to FAQ questions (WebFAQ) |
| Workflow | yes-or-no workflow decisions in Jev's format, on seen and unseen tasks |
| Politics | ideology and power in parliament speeches (Touché) |
| Language | inference (NLI-TR), sentence similarity (STSb-TR), reading (Belebele), common sense (XCOPA), grammar (TurBLiMP) |
| Knowledge | exam questions (INCLUDE, Global-MMLU, TurkishMMLU) |

## Scoring

Each question is shown with lettered options (A, B, C …), and the probability of an answer is the probability the model
gives its letter as the next token. Choice and Noul questions are asked a second time with the options reversed, and the
two answers are averaged. The tables report accuracy (how often the most likely answer is correct) and raw log-loss (the
average negative log-probability of the correct answer, before any calibration; lower is better). `results/scores.csv`
also holds macro-F1, the Brier score, the calibration error after a temperature fitted on dev, and the error on Score
questions.

## Results

Test split. Accuracy (%) on the 25 main tasks and by area, raw log-loss on the main tasks.

| Model | Accuracy | Raw log-loss | Routing | Opinion | Safety | Fact-checking | Judging | Workflow | Politics | Language | Knowledge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 77.6 | 0.81 | 87.4 | 62.4 | 90.6 | 51.2 | 81.7 | 87.3 | 86.8 | 78.9 | 86.9 |
| vistalab-system-one-12b | 77.6 | 0.59 | 88.8 | 65.0 | 92.9 | 54.5 | 83.3 | 89.3 | 82.0 | 80.4 | 74.1 |
| Microsoft-Decision-1 | 77.0 | 0.63 | 91.0 | 63.4 | 90.5 | 51.9 | 80.7 | 87.1 | 84.6 | 81.9 | 77.3 |
| tau-4b | 75.9 | 0.64 | 91.6 | 65.7 | 92.2 | 59.1 | 80.0 | 90.4 | 76.9 | 72.6 | 71.7 |
| Gemma 4 12B | 75.9 | 1.85 | 86.2 | 61.4 | 86.5 | 58.5 | 82.0 | 87.3 | 85.3 | 77.8 | 74.2 |
| vistalab-system-one-e4b | 75.4 | 0.62 | 87.8 | 63.3 | 92.1 | 57.5 | 79.5 | 89.7 | 76.6 | 76.3 | 66.4 |
| Qwen3.5 35B-A3B | 74.1 | 0.75 | 85.8 | 59.8 | 82.2 | 52.9 | 76.2 | 86.0 | 86.0 | 75.9 | 79.3 |
| Qwen3.5 9B | 70.6 | 0.80 | 81.1 | 54.5 | 80.6 | 56.4 | 77.7 | 84.2 | 76.7 | 70.1 | 70.5 |
| Trendyol LLM Asure 12B | 70.2 | 0.84 | 83.1 | 62.1 | 71.6 | 39.7 | 77.8 | 87.1 | 82.9 | 75.3 | 67.5 |
| Metask Jev 4B | 69.7 | 1.03 | – | 53.6 | 91.4 | 55.1 | 74.8 | 78.9 | 74.4 | 68.1 | 62.1 |
| Gemma 4 E4B | 68.9 | 1.61 | 83.2 | 59.1 | 81.7 | 44.4 | 72.0 | 86.1 | 80.2 | 69.1 | 63.0 |
| Turkish-Gemma 9B | 66.2 | 1.29 | 81.3 | 58.4 | 55.7 | 42.3 | 72.9 | 86.0 | 86.5 | 68.8 | 67.2 |
| Qwen3.5 4B | 65.9 | 0.96 | 75.9 | 49.9 | 67.7 | 52.1 | 78.9 | 82.9 | 64.1 | 67.9 | 63.8 |
| EuroLLM 9B | 61.4 | 1.06 | 75.0 | 52.0 | 71.2 | 43.0 | 65.1 | 84.2 | 73.6 | 60.1 | 47.9 |
| Kızagan E4B | 60.3 | 1.21 | 73.9 | 46.3 | 71.2 | 38.8 | 72.6 | 76.2 | 65.3 | 63.9 | 46.2 |
| ufakzeka-karar | 46.0 | 1.16 | 49.0 | 31.3 | 73.4 | 48.9 | 61.2 | 53.7 | 47.8 | 37.1 | 25.0 |
| Laya (multilingual) | 45.9 | 1.90 | 56.7 | 30.3 | 53.9 | 43.1 | 59.4 | 67.5 | 49.2 | 44.6 | 24.3 |
| Mecellem Qwen3 4B | 37.0 | 1.26 | 8.8 | 16.0 | 38.5 | 47.8 | 57.1 | 55.0 | 48.6 | 42.0 | 30.9 |
| Kumru 2B | 36.0 | 1.39 | 6.5 | 15.8 | 64.3 | 44.1 | 51.0 | 51.5 | 44.1 | 34.7 | 23.6 |

- Jev was evaluated through TypeSafe's API. Metask cannot answer MASSIVE, so it has no routing score.
- Microsoft-Decision-1 was evaluated through OpenRouter's Decisions API, which served it from Microsoft's Azure
  deployment. Microsoft post-trained it from Qwen3.5 9B and describes it as optimized for English.
- tau-4b (Cisimcik AI Lab) and ufakzeka-karar (ufak AI) [2] are open Turkish decision models. They were run on one GPU
  through their own `decide()` functions, one question per request as for Jev, and answered every question. tau-4b's
  probabilities are read before `decide()` rounds them to four decimals. Both were trained on Turkish classification
  data that may overlap with the benchmark's sources: ufakzeka-karar's model card lists MASSIVE, OffensEval 2020,
  MiDe22, FACTurk and WebFAQ among its training sets, and tau-4b's card describes its training data only in general
  terms.
- The two vistalab-system-one models are ours. They were not trained on any benchmark question. For a few tasks they
  were trained on other splits of the same datasets (the training splits of TrGLUE, whose validation and test splits
  make up the workflow tasks, and of Lavoir) or on other texts from the same sources (other WebFAQ sites, fact-check
  reports about other claims). A check found none of the benchmark's 106,003 dev and test texts in their training data.
- Small differences should be read with care: two training runs of the same 12B recipe scored 77.6 and 78.1.

## Running it

```bash
pip install -e .
s1bench build                                          # download and build the 66 tasks
s1bench run google/gemma-4-12b-it                      # a Hugging Face model, a local folder or a PEFT adapter, on one GPU
s1bench run jev                                        # Jev through TypeSafe's API
s1bench run openrouter/microsoft/microsoft-decision-1  # a decision model through OpenRouter's Decisions API
s1bench run cisimcik/tau-4b                            # an open decision model from Hugging Face, on one GPU
s1bench run ufakai/ufakzeka-karar                      # an open decision model from Hugging Face, on one GPU
s1bench score                                          # write results/scores.csv and print the tables
```

What `s1bench build` and `s1bench run` need:

- **Hugging Face:** one task, HateDay, uses a gated dataset, `manueltonneau/hateday`. Accept its terms on its Hugging
  Face page, then run `hf auth login` with a read token. The other Hugging Face datasets need no token.
- **Kaggle:** the SMS spam task downloads a public Kaggle dataset, `onurkarasoy/turkish-sms-collection`, with kagglehub.
  No account should be needed; if Kaggle asks for one, set `KAGGLE_API_TOKEN` to a token from your Kaggle settings.
- **Other sources:** some tasks download files from GitHub, GitLab, Zenodo and the NLI-TR and OffensEval pages, without
  an account.
- **Jev:** `s1bench run jev` needs a TypeSafe API key in `TYPESAFE_API_KEY`.
- **OpenRouter:** `s1bench run openrouter/<model>` runs a decision model such as Microsoft-Decision-1 through
  OpenRouter's Decisions API. It needs an OpenRouter API key in `OPENROUTER_API_KEY` and credit on the account.
- **Open decision models:** `s1bench run cisimcik/tau-4b` and `s1bench run ufakai/ufakzeka-karar` download the models
  from Hugging Face without a token and run code from their repositories: tau-4b's model code through
  `trust_remote_code`, and ufakzeka-karar's `karar.py`.
- **GPU:** local models run on one CUDA GPU in bfloat16; ufakzeka-karar runs in float32.

## Adding a task

A task is a function that downloads a dataset and returns its splits as lists of items.

1. Write the function in a module under `s1bench/tasks/`. Each item has a `state` (a string, or an object of named
   fields), one or more `questions`, and a `gold` answer for each question. `split` from `s1bench/split.py` samples a
   dev split of 200 items and a test split of 1,000 and keeps the rest; pass a dataset's own dev items as its second
   argument. `s1bench/download.py` has helpers for Hugging Face datasets (`hf_rows`, `hf_file`), Kaggle datasets
   (`kaggle_file`) and plain URLs (`download`).

   ```python
   from ..download import hf_rows
   from ..split import split

   QUESTION = {
       "type": "choice",
       "instructions": "Bu haber hangi kategoriye giriyor?",
       "criteria": {
           "spor": "Spor: Haber bir spor olayını ya da sporcuyu anlatıyor.",
           "ekonomi": "Ekonomi: Haber piyasaları, şirketleri ya da fiyatları anlatıyor.",
           "diğer": "Diğer: Haber bu kategorilerin hiçbirine girmiyor.",
       },
   }


   def build():
       items = [
           {"state": r["text"], "questions": {"topic": QUESTION}, "gold": {"topic": r["label"]}}
           for r in hf_rows("owner/dataset", None, "train")
       ]
       return split(items)
   ```

   A choice question maps each answer key to its criterion (or to `None`, to show the key itself); a noul question has
   `true` and `false` criteria and a gold of `true` or `false`; a score question lists its levels in order, and its
   gold is the level's index.
2. Register the function in `s1bench/tasks/__init__.py`: import its module at the top and add
   `"<name>_tr": <module>.build` to `TASKS`, or to `SECONDARY` for a secondary task. To count it in an area and in the
   main average, also add it to that area in `AREAS`; every model then needs a run on it.
3. Build and run it:

   ```bash
   s1bench build <name>_tr
   s1bench run google/gemma-4-12b-it <name>_tr
   s1bench score
   ```

The task data keep the licences of their original sources.

## References

1. S. F. Teke. HakemBench: A Turkish Benchmark of Typed Decisions. arXiv:2610.02293, 2026.
   [Paper](https://arxiv.org/abs/2610.02293), [dataset](https://huggingface.co/datasets/ufakai/HakemBench).
2. S. F. Teke. ufakzeka-karar: An Open Turkish Typed-Decision Model with Order-Invariant Option Scoring.
   arXiv:2610.06744, 2026. [Paper](https://arxiv.org/abs/2610.06744),
   [model](https://huggingface.co/ufakai/ufakzeka-karar).
