import json
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

from .api import answer_probabilities
from .prompt import options_of


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


def d3_folder():
    return snapshot_download("vllm-sr/d3", revision="62f07e54c6c3a5ff5f695107377edc43e1a86ea8")


def d3():
    model = AutoModel.from_pretrained(d3_folder(), trust_remote_code=True, dtype=torch.bfloat16, device="cuda:0")
    return LocalDecider(lambda state, questions: model.system_one(state=state, questions=questions))


def pusula():
    from pusula.inference import D3Engine

    adapter = snapshot_download("erenextlabs/pusula", revision="dfe77ba366c2a09049e118017c8f294f73a94634")
    # as released, Pusula answers a question outside its own templates with d3 alone; own_base=False keeps its adapter on
    engine = D3Engine.load(d3_folder(), adapter=adapter, device="cuda:0", load_in_4bit=False, own_base=False)

    def decide(state, questions):
        # the engine takes the text as a string; d3's own code turns a record into the same JSON
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        return {"answers": engine.decide(state, questions)}

    return LocalDecider(decide)


def laya_tr():
    folder = snapshot_download("TurkishCodeMan/laya-tr", revision="6b96146402db87d94148a68bfaf1a11b1a5e2fdf")
    tokenizer = AutoTokenizer.from_pretrained(folder, trust_remote_code=False)
    model = AutoModel.from_pretrained(folder, trust_remote_code=True).to("cuda").eval()
    # the rotary frequencies are not in the checkpoint, and transformers 5 leaves them unset; set them as the model's code does
    rotary = model.encoder.rotary_emb
    rotary.inv_freq = 1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32, device="cuda") / rotary.dim))
    # decide() rounds probabilities to four decimals; read them unrounded, as for the other open models
    sys.modules[type(model).__module__].round = lambda value, digits=None: value

    def decide(state, questions):
        (qid, question), = questions.items()
        options = options_of(question)
        # decide() takes one question, its options as a list of texts, and the text as a string
        context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=2)
        out = model.decide(question=question["instructions"], options=[text for _, text in options], tokenizer=tokenizer, context=context)
        probs = dict(zip([key for key, _ in options], out["probabilities"].values()))
        if question["type"] == "noul":
            return {"answers": {qid: {"type": "noul", "noul": probs["true"]}}}
        return {"answers": {qid: {"type": question["type"], "probabilities": probs}}}

    return LocalDecider(decide)


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
