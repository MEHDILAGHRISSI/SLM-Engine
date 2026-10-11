import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_benchmark(model_id: str, num_samples: int = 50):
    print(f"=== Running Baseline Benchmark for: {model_id} ===")
    
    # 1. Enable 4-bit quantization with CPU offloading for GPUs under 2 GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    # 2. Load dataset split
    print("Loading test dataset...")
    ds = load_dataset("ayshajavd/code-security-vulnerability-dataset", split=f"test[:{num_samples}]")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit model with CPU offloading...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    y_true = []
    y_pred = []
    latencies = []

    system_prompt = "Vous êtes un agent d'analyse statique de code. Répondez UNIQUEMENT par 'VULNERABLE' ou 'SAFE'."

    print(f"Running inference on {num_samples} samples...")
    for idx, sample in enumerate(ds):
        code_snippet = sample["code"]
        lang = sample["language"]
        ground_truth = 1 if sample["is_vulnerable"] else 0
        
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\nAnalysez la sécurité de ce code {lang} :\n```{lang}\n{code_snippet}\n```<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        
        with torch.no_grad():
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

    # 3. Compute benchmark metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    avg_latency = sum(latencies) / len(latencies)

    print("\n--- BASELINE BENCHMARK RESULTS ---")
    print(f"Model ID:              {model_id}")
    print(f"Evaluated Samples:     {num_samples}")
    print(f"Accuracy:              {acc * 100:.2f}%")
    print(f"Precision:             {precision:.4f}")
    print(f"Recall:                {recall:.4f}")
    print(f"F1-Score:              {f1:.4f}")
    print(f"Avg Latency / Sample:  {avg_latency * 1000:.2f} ms")

if __name__ == "__main__":
    run_benchmark("Qwen/Qwen2.5-Coder-3B-Instruct", num_samples=50)