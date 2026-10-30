"""QLoRA fine-tuning pipeline for TriageLM using Hugging Face PEFT and TRL."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train TriageLM with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="primary",
        choices=["primary", "ablation"],
        help="Model variant to train: primary (1.5B) or ablation (0.5B)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output checkpoint directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train():
    args = parse_args()
    cfg = load_config(args.config)

    model_info = cfg["models"][args.model_variant]
    model_id = model_info["name"]
    micro_batch = model_info["micro_batch_size"]
    grad_accum = model_info["gradient_accumulation_steps"]
    epochs = args.epochs or cfg["training"]["num_train_epochs"]
    out_dir = args.output_dir or f"{cfg['training']['output_dir']}/{args.model_variant}"

    print("=" * 60)
    print(f"Starting TriageLM Training: {model_id} [{args.model_variant.upper()}]")
    print(f"Batch Size: {micro_batch} | Grad Accum: {grad_accum} | Epochs: {epochs}")
    print(f"Checkpoints will be saved to: {out_dir}")
    print("=" * 60)

    # 1. Configure 4-bit NF4 Quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Tokenizer & Base Model
    print("Loading base model with 4-bit quantization...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to conserve VRAM on 6GB RTX 3050
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"]["gradient_checkpointing"]
    )

    # 3. Setup LoRA / PEFT Configuration
    peft_cfg = cfg["peft_lora"]
    lora_config = LoraConfig(
        r=peft_cfg["r"],
        lora_alpha=peft_cfg["lora_alpha"],
        lora_dropout=peft_cfg["lora_dropout"],
        bias=peft_cfg["bias"],
        task_type=peft_cfg["task_type"],
        target_modules=peft_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load Datasets
    print(f"Loading processed datasets from {cfg['data']['processed_dir']}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_file"],
            "validation": cfg["data"]["val_file"],
        },
    )

    def formatting_prompts_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(cfg["training"]["learning_rate"]),
        num_train_epochs=epochs,
        lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        weight_decay=cfg["training"]["weight_decay"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg["training"]["save_total_limit"],
        fp16=True,
        optim=cfg["training"]["optim"],
        gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
        report_to=cfg["training"]["report_to"],
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=cfg["training"]["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Beginning training loop...")
    train_result = trainer.train()

    print("Saving final adapter...")
    final_adapter_dir = f"{out_dir}/final_adapter"
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print(f"Training completed successfully! Adapter saved at: {final_adapter_dir}")
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)


if __name__ == "__main__":
    train()
