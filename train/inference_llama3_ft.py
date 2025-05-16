# inference_llama3_ft.py
#!/usr/bin/env python3
import argparse
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Inference for fine-tuned LLaMA-3-8B")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    # Load model + tokenizer
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    model.eval()
    while True:
        prompt = input("\n> ")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

