## 1. Choix du Dataset & Stratégie

Pour former un agent d'analyse de code complet et polyvalent, la meilleure approche consiste à séparer l'apprentissage de l'évaluation :

1. **Dataset d'Entraînement : `hitoshura25/crossvul`**
    - **Pourquoi ?** Il contient 9 310 fonctions réelles couvrant **21 langages** (Java, Python, JS, Go, C/C++, PHP) avec à la fois le code vulnérable, la description CWE et **le correctif exact (`fixed_code`)**.
    - C'est le dataset parfait pour apprendre au modèle 3B à **diagnostiquer ET corriger** les failles.
2. **Dataset de Test Indépendant : `ayshajavd/code-security-vulnerability-dataset`**
    - **Pourquoi ?** Il sert uniquement à l'évaluation finale hors-domaine. Il permettra d'évaluer la généralisation du modèle sur des échantillons 50/50 inédits.

## 2. Étape 1 : Préparation de l'Environnement sous Ubuntu (Terminal)

Sur la machine Ubuntu de votre coéquipière, ouvrir un terminal et exécuter :

Bash

```
# 1. Créer le dossier du projet
mkdir -p ~/projects/SLM_Agent
cd ~/projects/SLM_Agent

# 2. Créer et activer l'environnement virtuel Python
python3 -m venv venv
source venv/bin/activate

# 3. Mettre à jour pip et installer PyTorch avec support CUDA
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Installer les bibliothèques Hugging Face et QLoRA
pip install transformers peft trl bitsandbytes datasets accelerate
```

## 3. Étape 2 : Le Script de Fine-Tuning (`finetune_ubuntu_6gb.py`)

Ce script est spécifiquement calibré pour un **GPU de 6 Go de VRAM** :

- **Modèle :** `Qwen/Qwen2.5-Coder-3B-Instruct` (3 milliards de paramètres).
- **Quantification :** 4-bit NF4 via BitsAndBytes.
- **Optimisations VRAM :** `gradient_checkpointing=True`, `max_length=1536`, `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`.
- **Consommation VRAM estimée :** ~4.2 Go / 6 Go (marge de sécurité garantie contre les plantages OOM).

Créer le fichier :

Bash

```
nano finetune_ubuntu_6gb.py
```

Coller le code suivant :

Python

```
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. Configuration de la quantification 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"

print(f"=== [1/5] Chargement du modèle {model_id} (Quantification 4-bit) ===")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto"
)

# 2. Configuration LoRA (Ciblage des 7 projections)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. Chargement et filtrage du dataset CrossVul
print("=== [2/5] Chargement du dataset hitoshura25/crossvul ===")
raw_dataset = load_dataset("hitoshura25/crossvul", split="train")

TARGET_LANGS = ["java", "python", "py", "javascript", "js", "go", "php", "c", "cpp"]
MAX_CHAR_LENGTH = 3000  # Étendu à 3000 caractères grâce aux 6Go VRAM

filtered_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS
    and len(x["vulnerable_code"]) <= MAX_CHAR_LENGTH
    and len(x["fixed_code"]) <= MAX_CHAR_LENGTH
)

print(f"Nombre d'échantillons retenus après filtrage : {len(filtered_dataset)}")

# 4. Formatage au format ChatML (Paires Vulnérable + Sécurisé)
def format_prompts(batch):
    texts = []
    for vuln_code, fix_code, cwe, lang in zip(
        batch["vulnerable_code"], batch["fixed_code"], batch["cwe_id"], batch["language"]
    ):
        # Exemple 1 : Détection et proposition de patch
        text_vuln = (
            f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
            f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{vuln_code}\n```<|im_end|>\n"
            f"<|im_start|>assistant\nDIAGNOSTIC: Code Vulnérable ({cwe}).\nCorrection suggérée :\n```{lang}\n{fix_code}\n```<|im_end|>"
        )
        texts.append(text_vuln)

        # Exemple 2 : Validation de code sécurisé
        text_safe = (
            f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
            f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{fix_code}\n```<|im_end|>\n"
            f"<|im_start|>assistant\nDIAGNOSTIC: Code Sécurisé. Aucune vulnérabilité critique détectée.<|im_end|>"
        )
        texts.append(text_safe)

    return {"text": texts}

print("=== [3/5] Formatage du dataset au format ChatML ===")
formatted_dataset = filtered_dataset.map(
    format_prompts,
    batched=True,
    remove_columns=raw_dataset.column_names
)

# 5. Configuration SFT optimisée pour 6Go VRAM
sft_config = SFTConfig(
    dataset_text_field="text",
    max_length=1536,                 # Fenêtre de contexte de 1536 tokens
    per_device_train_batch_size=1,   # 1 exemple à la fois
    gradient_accumulation_steps=8,   # Accumulation sur 8 étapes (Batch size effectif = 8)
    max_steps=300,                   # 300 étapes pour une convergence solide
    learning_rate=2e-4,
    fp16=True,                       # Supporté nativement sur GPU Nvidia Ubuntu
    bf16=False,
    gradient_checkpointing=True,     # Réduit considérablement l'usage VRAM
    logging_steps=10,
    output_dir="outputs_qwen3b_ubuntu",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=sft_config,
)

# 6. Exécution du Fine-Tuning
print("=== [4/5] Démarrage du Fine-Tuning de Qwen2.5-Coder-3B ===")
trainer.train()

# 7. Sauvegarde des poids
print("=== [5/5] Sauvegarde du Modèle Fine-Tuné ===")
model.save_pretrained("outputs_qwen3b_ubuntu/qwen_3b_lora_final")
tokenizer.save_pretrained("outputs_qwen3b_ubuntu/qwen_3b_lora_final")
print("\n--- Fine-Tuning terminé avec succès ! ---")
```

## 4. Étape 3 : Exécution de l'Entraînement

Lancer le script dans le terminal Ubuntu :

Bash

```
python3 finetune_ubuntu_6gb.py
```

- **Durée estimée :** Environ **15 à 25 minutes** pour réaliser les 300 steps.
- Vous pouvez surveiller la mémoire GPU dans un second terminal avec la commande `watch -n 1 nvidia-smi`.

## 5. Étape 4 : Script d'Évaluation Équilibré (`eval_ubuntu_6gb.py`)

Une fois l'entraînement terminé, exécutez ce script de test sur le dataset indépendant `ayshajavd/code-security-vulnerability-dataset` (50 % vulnérable / 50 % sain) pour mesurer le gain de performance du modèle 3B.

Créer le fichier :

Bash

```
nano eval_ubuntu_6gb.py
```

Coller le code suivant :

Python

```
import torch
import time
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
lora_path = "outputs_qwen3b_ubuntu/qwen_3b_lora_final"

print("=== [1/3] Chargement du Modèle Qwen 2.5 Coder 3B + Adaptateur LoRA ===")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)
model.eval()

print("\n=== [2/3] Chargement du Dataset de Test Inédit (50/50 Équilibré) ===")
raw_dataset = load_dataset("ayshajavd/code-security-vulnerability-dataset", split="train")

TARGET_LANGS = ["Java", "Python", "JavaScript", "java", "python", "javascript", "C", "PHP"]
MAX_CHAR_LENGTH = 2000

vuln_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS
    and x["code"] is not None
    and len(x["code"]) <= MAX_CHAR_LENGTH
    and x["is_vulnerable"] is True
).select(range(50))

safe_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS
    and x["code"] is not None
    and len(x["code"]) <= MAX_CHAR_LENGTH
    and x["is_vulnerable"] is False
).select(range(50))

test_dataset = concatenate_datasets([vuln_dataset, safe_dataset]).shuffle(seed=42)

print("\n=== [3/3] Évaluation Zéro-Shot sur 100 Échantillons Inédits ===")

TP, FP, TN, FN = 0, 0, 0, 0
latencies = []

for idx, sample in enumerate(test_dataset, start=1):
    code = sample["code"]
    lang = sample["language"]
    is_vuln_ground_truth = sample["is_vulnerable"]

    prompt = (
        f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
        f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=False)
    latencies.append(time.time() - t0)

    reply = tokenizer.decode(out[0], skip_special_tokens=False)
    is_pred_vuln = "Code Vulnérable" in reply

    if is_vuln_ground_truth:
        if is_pred_vuln:
            TP += 1
        else:
            FN += 1
    else:
        if is_pred_vuln:
            FP += 1
        else:
            TN += 1

    if idx % 10 == 0:
        print(f"Progression : {idx}/100 faits...")

total = TP + FP + TN + FN
accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

print("\n" + "="*60)
print("     RÉSULTATS MODÈLE QWEN-3B FINE-TUNÉ (EVALUATION 50/50)")
print("="*60)
print(f"• Exactitude Globale (Accuracy) : {accuracy * 100:.2f} %")
print(f"• Précision (Fiabilité alerte)  : {precision * 100:.2f} %")
print(f"• Rappel (Sensibilité failles) : {recall * 100:.2f} %")
print(f"• Score F1                      : {f1 * 100:.2f} %")
print(f"• Taux de Faux Positifs (FPR)   : {fpr * 100:.2f} %")
print(f"• Latence Moyenne par Fonction  : {sum(latencies)/len(latencies):.2f} s")
print("="*60)
```

Lancer l'évaluation :

Bash

```
python3 eval_ubuntu_6gb.py
```

## 6. Étape 5 : Exporter pour Ollama (Pour tout le groupe)

Une fois le fine-tuning terminé, votre coéquipière peut fusionner le modèle et créer un fichier exportable pour qu'il fonctionne dans Ollama sur tous vos ordinateurs.

### Script de fusion (`export_to_ollama.py`) :

Python

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "Qwen/Qwen2.5-Coder-3B-Instruct"
lora_id = "outputs_qwen3b_ubuntu/qwen_3b_lora_final"

print("Chargement du modèle de base pour fusion...")
tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="cpu"  # Charge en RAM système pour la fusion
)

print("Fusion de l'adaptateurs LoRA...")
model = PeftModel.from_pretrained(base_model, lora_id)
merged_model = model.merge_and_unload()

print("Sauvegarde du modèle fusionné...")
merged_model.save_pretrained("qwen_coder_3b_merged")
tokenizer.save_pretrained("qwen_coder_3b_merged")
print("Fusion terminée avec succès dans 'qwen_coder_3b_merged' !")
```

Après exécution de `python3 export_to_ollama.py`, vous pouvez créer un `Modelfile` :

Dockerfile

```
FROM ./qwen_coder_3b_merged
PARAMETER temperature 0.2
SYSTEM "Vous êtes un agent expert en analyse statique de code et cybersécurité."
```

Et charger le modèle personnalisé dans Ollama :

Bash

```
ollama create code-analysis-agent -f Modelfile
```

Votre équipe disposera alors d'un modèle **`code-analysis-agent`** local exécutable directement via Ollama (`ollama run code-analysis-agent`).
