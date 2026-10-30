"""Adapter merger and export utility for TriageLM."""

import argparse
import os
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID of base model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="checkpoints/primary/final_adapter",
        help="Path to the saved LoRA adapter",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exports/merged_1.5b",
        help="Directory to save the merged model",
    )
    return parser.parse_args()


def merge_and_export():
    args = parse_args()

    print(f"Loading base model in fp16: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU to avoid GPU OOM during merging
        trust_remote_code=True,
    )

    print(f"Loading adapter weights from: {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter weights into base model...")
    merged_model = model.merge_and_unload()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged standalone model to: {output_path}...")
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)

    print("Export complete!")
    print("\nNext step: Convert to GGUF format for ultra-low latency CPU execution:")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_path} --outfile {output_path}/triagelm_q4_k_m.gguf --outtype q4_k_m")


if __name__ == "__main__":
    merge_and_export()
