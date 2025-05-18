import os
from transformers import LlamaForCausalLM, AutoTokenizer

# Path to training output dir
checkpoint_root = "output-h100"

# Find latest checkpoint (e.g. checkpoint-1944)
checkpoints = [
    d for d in os.listdir(checkpoint_root)
    if d.startswith("checkpoint-")
]
latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
latest_path = os.path.join(checkpoint_root, latest)

# Load the model and tokenizer
model = LlamaForCausalLM.from_pretrained(
    latest_path,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(latest_path, use_fast=True)

# Save in safetensors format
output_dir = "inference-llama3-8b"
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model exported to: {output_dir}")

