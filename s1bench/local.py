import json
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

from .api import answer_probabilities


class LocalDecider:
    def __init__(self, decide):
        self.decide = decide

    @torch.no_grad()
    def predict(self, items):
        predictions = []
        for item in items:
            answers = {}
            for qid, question in item["questions"].items():
                # a question the model does not take (too many levels, options it cannot tell apart) stays unanswered
                try:
                    answers[qid] = answer_probabilities(self.decide(item["state"], {qid: question})["answers"][qid])
                except ValueError:
                    answers[qid] = None
            predictions.append(answers)
        return predictions


def karar():
    folder = snapshot_download("ufakai/ufakzeka-karar")
    sys.path.insert(0, folder)
    from karar import Karar

    return LocalDecider(Karar.from_pretrained(folder, device="cuda").decide)


def tau():
    model = AutoModel.from_pretrained("cisimcik/tau-4b", trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
    # decide() rounds probabilities to four decimals; read them unrounded, as for the other open models
    sys.modules[type(model).__module__.replace("modeling_tau", "tau_schema")].round = lambda value, digits=None: value
    return LocalDecider(model.decide)


class Turnike:
    def __init__(self):
        folder = snapshot_download("berkay123001/turnike-jev-4b", revision="e65dc05482a4854c7122821db02ee04c63075bb6")
        sys.path.insert(0, folder)
        from karar import Karar

        self.karar = Karar(folder, device="cuda")

    def predict(self, items):
        predictions = []
        for item in items:
            # Karar takes the text as a string, and each question with only these keys
            state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"], ensure_ascii=False, indent=2)
            answers = {}
            for qid, question in item["questions"].items():
                asked = {key: question[key] for key in ("type", "instructions", "criteria") if key in question}
                try:
                    answers[qid] = self.karar.sor(state, {qid: asked})[qid]
                except ValueError:
                    answers[qid] = None
            predictions.append(answers)
        return predictions
