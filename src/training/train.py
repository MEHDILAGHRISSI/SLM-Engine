import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Configuration de la quantification 4-bit NF4 (optimisée NVIDIA T500)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Chargement du modèle de base Qwen2.5-Coder-0.5B...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto"
)

# 2. Configuration LoRA
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Chargement et filtrage du dataset crossvul
print("Chargement du dataset crossvul depuis Hugging Face...")
raw_dataset = load_dataset("hitoshura25/crossvul", split="train")

TARGET_LANGS = ["java", "python", "py", "javascript", "js", "go", "php", "c", "cpp"]
MAX_CHAR_LENGTH = 2000 

filtered_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS 
    and len(x["vulnerable_code"]) <= MAX_CHAR_LENGTH 
    and len(x["fixed_code"]) <= MAX_CHAR_LENGTH
)

print(f"Nombre d'échantillons retenus : {len(filtered_dataset)}")

# 4. Formatage au format ChatML
def format_prompts(batch):
    texts = []
    for vuln_code, fix_code, cwe, lang in zip(
        batch["vulnerable_code"], batch["fixed_code"], batch["cwe_id"], batch["language"]
    ):
        text_vuln = (
            f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
            f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{vuln_code}\n```<|im_end|>\n"
            f"<|im_start|>assistant\nDIAGNOSTIC: Code Vulnérable ({cwe}).\nCorrection suggérée :\n```{lang}\n{fix_code}\n```<|im_end|>"
        )
        texts.append(text_vuln)

        text_safe = (
            f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
            f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{fix_code}\n```<|im_end|>\n"
            f"<|im_start|>assistant\nDIAGNOSTIC: Code Sécurisé. Aucune vulnérabilité critique détectée.<|im_end|>"
        )
        texts.append(text_safe)

    return {"text": texts}

formatted_dataset = filtered_dataset.map(
    format_prompts, 
    batched=True,
    remove_columns=raw_dataset.column_names
)

# 5. Configuration d'entraînement SFT (max_length au lieu de max_seq_length)
sft_config = SFTConfig(
    dataset_text_field="text",
    max_length=1024,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=150,
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    logging_steps=10,
    output_dir="outputs_java_gpu",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=sft_config,
)

# 6. Exécution du Fine-Tuning
print("Démarrage du fine-tuning...")
trainer.train()

print("\n--- Fine-tuning GPU terminé avec succès ! ---")

# 7. Sauvegarde des poids LoRA
model.save_pretrained("outputs_java_gpu/qwen_coder_lora")
tokenizer.save_pretrained("outputs_java_gpu/qwen_coder_lora")
