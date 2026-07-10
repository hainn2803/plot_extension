import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcqa_constants import MODEL_NAME


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_gemma_model(device="cuda"):
    token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token, cache_dir=CACHE_DIR)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=token, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR)
    model.eval()
    return model, tokenizer
