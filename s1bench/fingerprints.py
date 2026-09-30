import hashlib
import json
import re

from .build import DATA


def words(text):
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return re.findall(r"[^\W_]+", re.sub(r"https?://\S+|@\w+", " ", text))


def fingerprint(words):
    return hashlib.blake2b(" ".join(words).encode(), digest_size=8).hexdigest()


def fingerprints(text):
    w = words(text)
    return {fingerprint(w)} | {fingerprint(w[i : i + 8]) for i in range(len(w) - 7)}


def texts(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for v in value for text in texts(v)]
    return []


def main():
    seen = set()
    for path in sorted(DATA.glob("*/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            for text in texts(json.loads(line)["state"]):
                if path.stem == "rest":
                    seen.add(fingerprint(words(text)))
                else:
                    seen |= fingerprints(text)
    (DATA / "fingerprints.txt").write_text("\n".join(sorted(seen)) + "\n")
    print(f"{len(seen)} fingerprints")


if __name__ == "__main__":
    main()
