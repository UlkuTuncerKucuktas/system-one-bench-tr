from .llm import LLM
from .system_one import Jev, Laya, Metask


def mecellem():
    model = LLM("newmindai/Mecellem-Qwen3-4B-TR")
    model.model.lm_head.weight = model.model.get_input_embeddings().weight  # the uploaded lm_head is all zeros
    return model


MODELS = {
    "jev": Jev,
    "metask-jev-4b": Metask,
    "laya-multilingual": Laya,
    "gemma-4-e4b": lambda: LLM("google/gemma-4-E4B-it"),
    "qwen3.5-4b": lambda: LLM("Qwen/Qwen3.5-4B"),
    "kizagan-e4b": lambda: LLM("AlicanKiraz0/Kizagan-E4B-Turkish-Reasoning-Model"),
    "mecellem-qwen3-4b": mecellem,
    "kumru-2b": lambda: LLM("vngrs/Kumru-2B"),
    "qwen3.5-9b": lambda: LLM("Qwen/Qwen3.5-9B"),
    "gemma-4-12b": lambda: LLM("google/gemma-4-12b-it"),
    "turkish-gemma-9b": lambda: LLM("ytu-ce-cosmos/Turkish-Gemma-9b-v0.1"),
    "trendyol-asure-12b": lambda: LLM("Trendyol/Trendyol-LLM-Asure-12B"),
    "eurollm-9b": lambda: LLM("utter-project/EuroLLM-9B-Instruct-2512"),
    "qwen3.5-35b-a3b": lambda: LLM("Qwen/Qwen3.5-35B-A3B"),
}
