# scripts/inference_cli.py
#!/usr/bin/env python3
import argparse
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Command-line querying of fine-tuned LLaMA-3-8B models"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--prompt", type=str, help="Prompt text to generate from"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Number of tokens to generate beyond the prompt"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50,
        help="Top-K sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95,
        help="Top-p (nucleus) sampling"
    )
    args = parser.parse_args()

    # Read prompt from stdin if not provided
    if args.prompt is None:
        import sys
        args.prompt = sys.stdin.read().strip()
        if not args.prompt:
            parser.error("No prompt provided via --prompt or stdin.")

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode and print only the generated continuation
    generated = outputs[0][inputs['input_ids'].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(text)

if __name__ == "__main__":
    main()

