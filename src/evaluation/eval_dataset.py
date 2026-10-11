import torch
import time
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------
# 1. Chargement du modèle fine-tuné
# ---------------------------------------------------------
base_model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
lora_path = "outputs_java_gpu/checkpoint-150"

print("=== [1/3] Chargement du Modèle Qwen 2.5 Coder + Adaptateur LoRA ===")
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

# ---------------------------------------------------------
# 2. Chargement & Équilibrage Stricte 50/50 du Dataset Indépendant
# ---------------------------------------------------------
print("\n=== [2/3] Chargement & Équilibrage 50/50 (ayshajavd/code-security-vulnerability-dataset) ===")
raw_dataset = load_dataset("ayshajavd/code-security-vulnerability-dataset", split="train")

TARGET_LANGS = ["Java", "Python", "JavaScript", "java", "python", "javascript", "C", "PHP"]
MAX_CHAR_LENGTH = 1500

# 1. Sélection de 50 exemples réellement VULNÉRABLES
vuln_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS 
    and x["code"] is not None 
    and len(x["code"]) <= MAX_CHAR_LENGTH 
    and x["is_vulnerable"] is True
).select(range(50))

# 2. Sélection de 50 exemples réellement SAINS
safe_dataset = raw_dataset.filter(
    lambda x: x["language"] in TARGET_LANGS 
    and x["code"] is not None 
    and len(x["code"]) <= MAX_CHAR_LENGTH 
    and x["is_vulnerable"] is False
).select(range(50))

# 3. Fusion et mélange aléatoire des 100 échantillons (50 Vulnérables / 50 Sains)
test_dataset = concatenate_datasets([vuln_dataset, safe_dataset]).shuffle(seed=42)

print(f"Nombre d'échantillons équilibrés chargés : {len(test_dataset)} (50 Vulnérables / 50 Sains)")

# ---------------------------------------------------------
# 3. Évaluation Zéro-Shot
# ---------------------------------------------------------
print("\n=== [3/3] Évaluation en cours sur le Jeu d'Épreuve Équilibré... ===")

TP = 0  # Vrais Positifs (Vulnérable bien détecté)
FP = 0  # Faux Positifs (Sain détecté comme vulnérable)
TN = 0  # Vrais Négatifs (Sain bien détecté)
FN = 0  # Faux Négatifs (Vulnérable manqué)
latencies = []

for idx, sample in enumerate(test_dataset, start=1):
    code = sample["code"]
    lang = sample["language"]
    is_vuln_ground_truth = sample["is_vulnerable"]  # True ou False

    prompt = (
        f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
        f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, temperature=0.1, do_sample=False)
    latencies.append(time.time() - t0)

    reply = tokenizer.decode(out[0], skip_special_tokens=False)
    is_pred_vuln = "Code Vulnérable" in reply

    # Matrice de confusion
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
        print(f"Progression : {idx}/{len(test_dataset)} échantillons évalués...")

# ---------------------------------------------------------
# 4. Calcul et Affichage des Métriques Non Biaisées
# ---------------------------------------------------------
total = TP + FP + TN + FN
accuracy = (TP + TN) / total if total > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
avg_latency = sum(latencies) / len(latencies)

print("\n" + "="*60)
print("     RÉSULTATS DE L'ÉVALUATION ÉQUILIBRÉE 50/50 (JEU INÉDIT)")
print("="*60)
print(f"Source du test : ayshajavd/code-security-vulnerability-dataset")
print(f"Échantillons : {total} (50 Vrais Vulnérables / 50 Vrais Sains)")
print(f"• Vrais Positifs (TP)  : {TP}")
print(f"• Vrais Négatifs (TN) : {TN}")
print(f"• Faux Positifs (FP)  : {FP}")
print(f"• Faux Négatifs (FN)  : {FN}")
print("-" * 60)
print(f"• Exactitude Globale (Accuracy) : {accuracy * 100:.2f} %")
print(f"• Précision (Fiabilité)         : {precision * 100:.2f} %")
print(f"• Rappel (Sensibilité)          : {recall * 100:.2f} %")
print(f"• Score F1                      : {f1 * 100:.2f} %")
print(f"• Taux de Faux Positifs (FPR)   : {fpr * 100:.2f} %")
print(f"• Temps moyen par fonction      : {avg_latency:.2f} s")
print("="*60)
