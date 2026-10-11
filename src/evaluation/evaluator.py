import os
import gc
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DATASET_NAME = "ayshajavd/code-security-vulnerability-dataset"

# Comprehensive Benchmark Suite
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "google/gemma-2-2b-it",
    "google/codegemma-7b-it",
]

def evaluate_model(model_id: str, dataset_samples, hf_token: str, max_seq_len: int = 2048):
    print(f"\n==================================================")
    print(f"  Evaluating Model: {model_id}")
    print(f"==================================================")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    except Exception as e:
        print(f"Error loading tokenizer for {model_id}: {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading weights onto GPU (FP16)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
    except Exception as e:
        print(f"Error loading model weights for {model_id}: {e}")
        return None

    y_true = []
    y_pred = []
    latencies = []

    system_prompt = "Vous êtes un agent d'analyse statique de code. Répondez UNIQUEMENT par 'VULNERABLE' ou 'SAFE'."

    print(f"Running inference on {len(dataset_samples)} samples...")
    for sample in dataset_samples:
        code_snippet = sample["code"]
        lang = sample["language"]
        ground_truth = 1 if sample["is_vulnerable"] else 0
        
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code_snippet}\n```"}
        ]
        
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback ChatML format if chat_template is not present
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code_snippet}\n```<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

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

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    avg_latency = (sum(latencies) / len(latencies)) * 1000

    # Strict VRAM cleanup between model runs
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
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", None)
    
    print(f"Loading '{DATASET_NAME}' split test[:{args.samples}]...")
    test_ds = load_dataset(DATASET_NAME, split=f"test[:{args.samples}]")

    results = []
    for model_id in DEFAULT_MODELS:
        res = evaluate_model(model_id, test_ds, hf_token)
        if res:
            results.append(res)

    print("\n" + "="*85)
    print(f"               FULL SLM COMPARATIVE LEADERBOARD ({args.samples} SAMPLES)")
    print("="*85)
    print(f"{'Model ID':<38} | {'Acc (%)':<8} | {'Prec':<7} | {'Recall':<7} | {'F1-Score':<8} | {'Latency':<9}")
    print("-" * 85)
    for r in results:
        print(f"{r['model_id']:<38} | {r['accuracy']:<8.2f} | {r['precision']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<8.4f} | {r['latency_ms']:.1f} ms")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()