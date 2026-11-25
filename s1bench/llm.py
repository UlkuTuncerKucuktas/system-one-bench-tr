import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .prompt import LABELS, label_token_ids, orders_of, render


class LLM:
    batch_size = 8

    def __init__(self, address):
        adapter = Path(address, "adapter_config.json").exists()
        repo = PeftConfig.from_pretrained(address).base_model_name_or_path if adapter else address
        config = AutoConfig.from_pretrained(repo)
        # Gemma 4 and Qwen3.5 are image-text models, and an adapter trained on one only fits that class
        model_class = AutoModelForImageTextToText if hasattr(config, "vision_config") else AutoModelForCausalLM
        self.model = model_class.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
        if adapter:
            self.model = PeftModel.from_pretrained(self.model, address).merge_and_unload().eval()
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
        # a question with more options than letters is skipped (jev-bench-tr's banking77, clinc150 and ledgar)
        jobs = [
            (item, qid, options)
            for item in items
            for qid, question in item["questions"].items()
            if len(question["criteria"]) <= len(LABELS)
            for options in orders_of(question)
        ]
        texts = [self.prompt(render(item, item["questions"][qid], options)) for item, qid, options in jobs]
        by_length = sorted(range(len(jobs)), key=lambda j: len(texts[j]))
        probabilities = [None] * len(jobs)
        for start in range(0, len(jobs), self.batch_size):
            batch = by_length[start : start + self.batch_size]
            results = self.label_probabilities([texts[j] for j in batch], [len(jobs[j][2]) for j in batch])
            for j, probs in zip(batch, results):
                probabilities[j] = probs

        sums = {}
        for (item, qid, options), probs in zip(jobs, probabilities):
            votes = sums.setdefault(item["id"], {}).setdefault(qid, {})
            for (key, _), p in zip(options, probs):
                votes[key] = votes.get(key, 0) + p
        return [
            {qid: {key: v / sum(votes.values()) for key, v in votes.items()} for qid, votes in sums.get(item["id"], {}).items()}
            for item in items
        ]


class VLLM(LLM):
    # a model that needs more than one GPU, through vLLM on every GPU, with the same prompts, letters and averaging
    batch_size = 4096
    settings = {"max_num_seqs": 128}

    def __init__(self, address):
        from vllm import LLM as Engine
        from vllm import SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(address)
        self.label_ids = [label_token_ids(self.tokenizer, label) for label in LABELS]
        self.engine = Engine(address, dtype="bfloat16", tensor_parallel_size=torch.cuda.device_count(), max_model_len=32768, gpu_memory_utilization=0.94, language_model_only=True, **self.settings)
        # the letters' log-probabilities over the whole vocabulary, not only among the most probable tokens
        letter_ids = sorted({i for ids in self.label_ids for i in ids})
        self.params = SamplingParams(max_tokens=1, temperature=0, logprob_token_ids=letter_ids, detokenize=False)

    def label_probabilities(self, texts, sizes):
        from vllm.inputs import TokensPrompt

        prompts = [TokensPrompt(prompt_token_ids=self.tokenizer(text, add_special_tokens=False).input_ids) for text in texts]
        rows = [output.outputs[0].logprobs[0] for output in self.engine.generate(prompts, self.params)]
        return [
            torch.stack([torch.tensor([row[i].logprob for i in ids]).logsumexp(0) for ids in self.label_ids[:size]]).softmax(0).tolist()
            for row, size in zip(rows, sizes)
        ]


class DeepSeek(VLLM):
    # DeepSeek-V4 has no chat template, only the encoder in its repository; its start-up on two GPUs needs smaller batches
    settings = {"max_num_seqs": 32, "max_num_batched_tokens": 4096}

    def __init__(self):
        address = "deepseek-ai/DeepSeek-V4-Flash"
        sys.path.insert(0, str(Path(snapshot_download(address, allow_patterns="encoding/*"), "encoding")))
        from encoding_dsv4 import encode_messages

        self.encode = encode_messages
        super().__init__(address)

    def prompt(self, text):
        return self.encode([{"role": "user", "content": text}], thinking_mode="chat")
