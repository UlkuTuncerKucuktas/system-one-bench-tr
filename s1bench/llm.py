import json
import string

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits
BATCH_SIZE = 8


def options_of(question):
    criteria = question["criteria"]
    if question["type"] == "choice":
        return [(key, text or key) for key, text in criteria.items()]
    if question["type"] == "noul":
        return [("true", criteria["true"]), ("false", criteria["false"])]
    return [(str(level), text) for level, text in enumerate(criteria)]


def orders_of(question):
    options = options_of(question)
    if question["type"] == "score":
        return [options]
    return [options, options[::-1]]


def render(item, question, options):
    state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"], ensure_ascii=False, indent=2)
    lines = [f"{label}) {text}" for label, (_, text) in zip(LABELS, options)]
    return f"{state}\n\n{question['instructions']}\n\n" + "\n".join(lines) + "\n\nYalnızca doğru seçeneğin harfini yaz."


def label_token_ids(tokenizer, label):
    ids = {tokenizer.convert_tokens_to_ids(token) for token in [label, "▁" + label, "Ġ" + label]}
    return [i for i in ids if i is not None and i != tokenizer.unk_token_id]


class LLM:
    def __init__(self, repo):
        config = AutoConfig.from_pretrained(repo)
        model_class = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
        self.model = model_class.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(repo, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_ids = [label_token_ids(self.tokenizer, label) for label in LABELS]

    def prompt(self, text):
        if self.tokenizer.chat_template is None:
            return text + "\n\nCevap:"
        messages = [{"role": "user", "content": text}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def label_probabilities(self, texts, sizes):
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        logits = self.model(**batch, logits_to_keep=1).logits[:, -1].float()
        return [
            torch.stack([row[ids].logsumexp(0) for ids in self.label_ids[:size]]).softmax(0).tolist()
            for row, size in zip(logits, sizes)
        ]

    def predict(self, items):
        jobs = [
            (item, qid, options)
            for item in items
            for qid, question in item["questions"].items()
            for options in orders_of(question)
        ]
        texts = [self.prompt(render(item, item["questions"][qid], options)) for item, qid, options in jobs]
        by_length = sorted(range(len(jobs)), key=lambda j: len(texts[j]))
        probabilities = [None] * len(jobs)
        for start in range(0, len(jobs), BATCH_SIZE):
            batch = by_length[start : start + BATCH_SIZE]
            results = self.label_probabilities([texts[j] for j in batch], [len(jobs[j][2]) for j in batch])
            for j, probs in zip(batch, results):
                probabilities[j] = probs

        sums = {}
        for (item, qid, options), probs in zip(jobs, probabilities):
            votes = sums.setdefault(item["id"], {}).setdefault(qid, {})
            for (key, _), p in zip(options, probs):
                votes[key] = votes.get(key, 0) + p
        return [
            {qid: {key: v / sum(votes.values()) for key, v in votes.items()} for qid, votes in sums[item["id"]].items()}
            for item in items
        ]
