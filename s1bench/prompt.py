import json
import string

LABELS = string.ascii_uppercase + string.ascii_lowercase + string.digits


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
