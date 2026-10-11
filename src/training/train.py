import os
import yaml
import argparse
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import Trainer

class LoRAPlusSFTTrainer(SFTTrainer):
    """SFTTrainer override implementing the LoRA+ optimizer parameter grouping."""
    def __init__(self, *args, loraplus_lr_ratio: float = 16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loraplus_lr_ratio = loraplus_lr_ratio

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            param_groups = {"lora_a": [], "lora_b": [], "other": []}
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "lora_B" in name or "lora_embedding_B" in name:
                    param_groups["lora_b"].append(param)
                elif "lora_A" in name or "lora_embedding_A" in name:
                    param_groups["lora_a"].append(param)
                else:
                    param_groups["other"].append(param)

            base_lr = self.args.learning_rate
            ratio_lr = base_lr * self.loraplus_lr_ratio

            optimizer_grouped_parameters = [
                {"params": param_groups["lora_a"], "lr": base_lr, "weight_decay": self.args.weight_decay},
                {"params": param_groups["lora_b"], "lr": ratio_lr, "weight_decay": self.args.weight_decay},
            ]
            if param_groups["other"]:
                optimizer_grouped_parameters.append(
                    {"params": param_groups["other"], "lr": base_lr, "weight_decay": self.args.weight_decay}
                )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
        return self.optimizer

def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("=== Initializing FastLanguageModel with Unsloth ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["name_or_path"],
        max_seq_length=cfg["model"]["max_seq_length"],
        load_in_4bit=cfg["model"]["load_in_4bit"],
        dtype=None,
    )

    print("=== Applying LoRA Adapters ===")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"]["target_modules"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print(f"=== Loading Fine-Tuning Dataset ({cfg['training']['dataset_name']}) ===")
    ds = load_dataset(cfg["training"]["dataset_name"], split="train")

    def format_prompts(batch):
        formatted = []
        for v_code, f_code, cwe, lang in zip(
            batch["vulnerable_code"], batch["fixed_code"], batch["cwe_id"], batch["language"]
        ):
            # Formulating SFT Instruction Pairs
            prompt = (
                f"<|im_start|>system\nVous êtes un agent d'analyse statique et de sécurisation de code.<|im_end|>\n"
                f"<|im_start|>user\nExaminez ce code {lang} et identifiez s'il est vulnérable ({cwe}) :\n```{lang}\n{v_code}\n```<|im_end|>\n"
                f"<|im_start|>assistant\nVULNERABLE\nCorrective Patch:\n```{lang}\n{f_code}\n```<|im_end|>"
            )
            formatted.append(prompt)
        return {"text": formatted}

    dataset = ds.map(format_prompts, batched=True)

    sft_config = SFTConfig(
        output_dir=cfg["training"]["output_dir"],
        dataset_text_field="text",
        max_seq_length=cfg["model"]["max_seq_length"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        optim="adamw_8bit",
        report_to="none",
    )

    trainer_cls = LoRAPlusSFTTrainer if cfg.get("loraplus", {}).get("enabled", False) else SFTTrainer
    trainer_kwargs = {"loraplus_lr_ratio": cfg["loraplus"].get("lr_ratio", 16.0)} if cfg.get("loraplus", {}).get("enabled", False) else {}

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        **trainer_kwargs
    )

    print("=== Starting Fine-Tuning with LoRA+ ===")
    trainer.train()

    print("=== Exporting Fine-Tuned Model to 4-Bit GGUF ===")
    model.save_pretrained_gguf(
        f"{cfg['training']['output_dir']}-GGUF",
        tokenizer,
        quantization_method="q4_k_m"
    )
    print("Training and GGUF export complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    train(args.config)