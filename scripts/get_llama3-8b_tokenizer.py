# from your repo root
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",  # or whatever HF ID you’re using
    use_fast=True
)
tok.save_pretrained(".")       # writes tokenizer.json + related files here
