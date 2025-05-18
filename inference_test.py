from transformers import AutoTokenizer, LlamaForCausalLM
import torch

# Load your exported model
model_path = "inference-llama3-8b"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Prompt (adapt to your fine-tuning style)
prompt = "Q: What is the capital of France?\nReasoning:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
)

print("🧠 Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
