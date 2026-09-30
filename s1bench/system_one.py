import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import torch

from .download import download


def answer_probabilities(answer):
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


class Jev:
    def __init__(self):
        self.client = httpx.Client(headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}"}, timeout=60)

    def predict(self, items):
        # one question per request, as for the other models: which questions an item has can depend on its labels
        jobs = [(item, qid) for item in items for qid in item["questions"]]
        with ThreadPoolExecutor(16) as pool:
            answers = list(pool.map(self.ask, jobs))
        predictions = {item["id"]: {} for item in items}
        for (item, qid), answer in zip(jobs, answers):
            predictions[item["id"]][qid] = answer
        return [predictions[item["id"]] for item in items]

    def ask(self, job):
        item, qid = job
        body = {"model": "jev-latest", "state": item["state"], "questions": {qid: item["questions"][qid]}}
        for attempt in range(6):
            try:
                response = self.client.post("https://api.typesafe.ai/v1/systemone", json=body)
                if response.status_code != 429 and response.status_code < 500:
                    response.raise_for_status()
                    return answer_probabilities(response.json()["answers"][qid])
            except httpx.TransportError:
                pass
            time.sleep(2**attempt)
        raise RuntimeError(f"Jev kept failing on {item['id']} {qid}")


class Laya:
    def __init__(self):
        import laya

        self.agent = laya.load("convaiinnovations/laya", subfolder="multilingual")

    def predict(self, items):
        return [
            {qid: answer_probabilities(self.agent.predict(item["state"], {qid: q})["answers"][qid]) for qid, q in item["questions"].items()}
            for item in items
        ]


class Metask:
    def __init__(self):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        sys.path.append(str(download("https://raw.githubusercontent.com/metask-ai/metask-jev/main/inference/jev_schema.py").parent))
        repo = "wayfind/metask-jev-4b-policy-mix"
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(repo, dtype=torch.bfloat16, device_map="cuda").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(repo)

    @torch.no_grad()
    def probabilities(self, state, question):
        from jev_schema import choice_key, prepare_prompts

        criteria = question["criteria"]
        if question["type"] == "choice":
            if len(criteria) > 26:
                return None
            field = {"type": "enum", "choices": list(criteria), "choice_descriptions": {k: v for k, v in criteria.items() if v}}
        elif question["type"] == "noul":
            field = {"type": "boolean", "choice_descriptions": criteria}
        else:
            levels = [str(level) for level in range(len(criteria))]
            field = {"type": "enum", "choices": levels, "choice_descriptions": dict(zip(levels, criteria))}
        field["description"] = question["instructions"]

        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        # the other models read the whole state, so Metask does too instead of refusing prompts over 4,096 tokens
        prepared = prepare_prompts(self.tokenizer, context, {"decision": field}, max_input_tokens=32768)
        ids = torch.tensor([prepared.full_ids[0]], device="cuda")
        logits = self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        probs = logits[prepared.candidate_ids[0]].softmax(-1).tolist()
        return {choice_key(choice): p for choice, p in zip(prepared.choices[0], probs)}

    def predict(self, items):
        return [{qid: self.probabilities(item["state"], q) for qid, q in item["questions"].items()} for item in items]
