#!/usr/bin/env python3
"""
SLM-Engine: Fine-tuning Qwen2.5-Coder for C/C++ Static Vulnerability Detection
"""

import argparse
import hashlib
import os
import sys
import warnings
import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Masquage des avertissements secondaires
warnings.filterwarnings("ignore", category=UserWarning)


def load_config(config_path: str) -> dict:
    """Charge le fichier de configuration YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_chatml(row: pd.Series) -> str:
    """
    Formatage des données au format ChatML avec prompt système aligné sur la classification.
    """
    verdict = "VERDICT: VULNERABLE" if row["vul"] == 1 else "VERDICT: SAFE"
    return (
        "<|im_start|>system\n"
        "You are a C/C++ static code analysis auditor.\n"
        "Classify the provided function snippet strictly as SAFE or VULNERABLE.\n"
        "RULES:\n"
        "1. Base your assessment ONLY on the visible code.\n"
        "2. Conclude strictly with either 'VERDICT: SAFE' or 'VERDICT: VULNERABLE'.<|im_end|>\n"
        "<|im_start|>user\n"
        f"Audit this code snippet:\n```cpp\n{row['func']}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{verdict}<|im_end|>"
    )


def prepare_datasets(dataset_name: str, max_func_len: int = 4000):
    """
    Pipeline de nettoyage : Déduplication par SHA-256 (Anti-leakage), 
    Filtrage par longueur et équilibrage strict 1:1.
    """
    print(f"\n=== 1. Chargement & Nettoyage du Dataset : {dataset_name} ===")
    
    try:
        raw_train = load_dataset(dataset_name, split="train").to_pandas()
        raw_test = load_dataset(dataset_name, split="test").to_pandas()
    except Exception as e:
        print(f"[!] Avertissement lors du chargement des splits ({e}). Chargement global par défaut.")
        full_ds = load_dataset(dataset_name, split="train").to_pandas()
        from sklearn.model_selection import train_test_split
        raw_train, raw_test = train_test_split(full_ds, test_size=0.2, random_state=42)

    # 1. Déduplication SHA-256 contre le jeu de test (anti-data-leakage)
    raw_train["code_hash"] = raw_train["func"].apply(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest())
    raw_test["code_hash"] = raw_test["func"].apply(lambda x: hashlib.sha256(str(x).encode("utf-8")).hexdigest())
    
    test_hashes = set(raw_test["code_hash"])
    clean_train = raw_train[~raw_train["code_hash"].isin(test_hashes)].copy()
    print(f"[-] {len(raw_train) - len(clean_train)} échantillons en double/fuites supprimés du train set.")

    # 2. Filtrage des fonctions trop longues
    clean_train = clean_train[clean_train["func"].str.len() <= max_func_len]

    # 3. Équilibrage strict 1:1 (Safe vs Vulnerable)
    vuln_df = clean_train[clean_train["vul"] == 1]
    safe_df = clean_train[clean_train["vul"] == 0]
    min_samples = min(len(vuln_df), len(safe_df))
    
    balanced_train = pd.concat([
        vuln_df.sample(n=min_samples, random_state=42),
        safe_df.sample(n=min_samples, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"[+] Dataset d'entraînement équilibré : {len(balanced_train)} échantillons ({min_samples} Vuln / {min_samples} Safe)")

    # 4. Jeu de validation pour le suivi
    clean_val = raw_test[raw_test["func"].str.len() <= max_func_len].sample(
        n=min(300, len(raw_test)), random_state=42
    ).reset_index(drop=True)

    # 5. Application du template ChatML
    balanced_train["text"] = balanced_train.apply(format_chatml, axis=1)
    clean_val["text"] = clean_val.apply(format_chatml, axis=1)

    return Dataset.from_pandas(balanced_train[["text"]]), Dataset.from_pandas(clean_val[["text"]])


def main():
    parser = argparse.ArgumentParser(description="Moteur de Fine-Tuning SLM pour la détection de vulnérabilités.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Chemin du fichier config.yaml")
    args = parser.parse_args()

    # Charger la configuration
    config = load_config(args.config)

    # Importations Unsloth locales
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # 1. Traitement des données
    train_dataset, val_dataset = prepare_datasets(config["training"]["dataset_name"])

    # 2. Chargement du modèle & Adaptateurs LoRA
    print("\n=== 2. Initialisation du modèle et de LoRA ===")
    max_seq_length = config["model"].get("max_seq_length", 2048)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name_or_path"],
        max_seq_length=max_seq_length,
        load_in_4bit=config["model"].get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["lora_alpha"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # 3. Configuration de l'entraînement
    print("\n=== 3. Configuration de SFTTrainer ===")
    output_dir = config["training"]["output_dir"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"].get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 4),
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"].get("num_train_epochs", 1),
        logging_steps=config["training"].get("logging_steps", 10),
        save_strategy=config["training"].get("save_strategy", "steps"),
        save_steps=config["training"].get("save_steps", 100),
        save_total_limit=config["training"].get("save_total_limit", 2),
        eval_strategy="steps",
        eval_steps=config["training"].get("eval_steps", 50),
        fp16=config["training"].get("fp16", not is_bfloat16_supported()),
        bf16=config["training"].get("bf16", is_bfloat16_supported()),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Application du masque de perte (Response-Only Loss)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # 4. Lancement du Fine-Tuning
    print("\n=== 4. Démarrage de l'entraînement ===")
    trainer.train()

    # 5. Exportation du modèle fusionné (16-bit)
    final_export_path = os.path.join(output_dir, "final_merged_f16")
    print(f"\n=== 5. Exportation du modèle fusionné dans : {final_export_path} ===")
    
    model.save_pretrained_merged(
        final_export_path,
        tokenizer,
        save_method="merged_16bit"
    )

    print("\n[✔] Entraînement et exportation réussis !")


if __name__ == "__main__":
    main()