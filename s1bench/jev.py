import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


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
