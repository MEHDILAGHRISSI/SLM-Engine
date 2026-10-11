import os
import gc
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Dataset d'évaluation fixe
DATASET_NAME = "ayshajavd/code-security-vulnerability-dataset"

def evaluate_model(model_id: str, dataset_samples, hf_token: str, max_seq_len: int = 2048):
    print(f"\n==================================================")
    print(f"  Evaluating Model: {model_id}")
    print(f"==================================================")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading weights onto GPU (FP16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )

    y_true = []
    y_pred = []
    latencies = []

    print(f"Running inference on {len(dataset_samples)} samples...")
    for sample in dataset_samples:
        code_snippet = sample["code"]
        lang = sample["language"]
        ground_truth = 1 if sample["is_vulnerable"] else 0
        
        messages = [
            {"role": "user", "content": f"Vous êtes un agent d'analyse statique de code. Répondez UNIQUEMENT par 'VULNERABLE' ou 'SAFE'.\n\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code_snippet}\n```"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=max_seq_len, 
            truncation=True
        ).to(model.device)

        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False
            )
        end_time = time.time()
        latencies.append(end_time - start_time)

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
        prediction = 1 if "VULNERABLE" in generated_text else 0
        
        y_true.append(ground_truth)
        y_pred.append(prediction)

    # Calcul des métriques
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    avg_latency = (sum(latencies) / len(latencies)) * 1000  # ms

    # Nettoyage mémoire VRAM pour libérer le GPU avant le modèle suivant
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "accuracy": acc * 100,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": avg_latency
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000, help="Nombre d'échantillons de test")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    
    # 1. Chargement unique du dataset pour garantir des données 100% identiques
    print(f"Loading '{DATASET_NAME}' split test[:{args.samples}]...")
    test_ds = load_dataset(DATASET_NAME, split=f"test[:{args.samples}]")

    models_to_benchmark = [
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "google/gemma-2-2b-it"
    ]

    results = []
    for model_id in models_to_benchmark:
        res = evaluate_model(model_id, test_ds, hf_token)
        results.append(res)

    # 2. Tableau comparatif synthétique
    print("\n" + "="*80)
    print(f"          COMPARATIVE BENCHMARK SUMMARY ({args.samples} SAMPLES)")
    print("="*80)
    print(f"{'Model ID':<35} | {'Acc (%)':<8} | {'Prec':<7} | {'Recall':<7} | {'F1-Score':<8} | {'Latency':<9}")
    print("-" * 80)
    for r in results:
        print(f"{r['model_id']:<35} | {r['accuracy']:<8.2f} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<8.4f} | {r['latency_ms']:.1f} ms")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()