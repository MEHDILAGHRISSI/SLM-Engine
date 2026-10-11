import time
import ollama

prompts = [
    "Write a Python function to validate an email address using regex.",
    "Identify the bug in this code: result = 10 / 0"
]

models = ["qwen2.5-coder:3b", "phi3.5"]

for model in models:
    print(f"\n================ TEST : {model} ================")
    for p in prompts:
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': p}])
        
        # Extraction des métriques de tokens fournies par Ollama
        prompt_tokens = response.get('prompt_eval_count', 0)  # Tokens d'entrée
        eval_tokens = response.get('eval_count', 0)            # Tokens de sortie (générés)
        eval_duration_sec = response.get('eval_duration', 1) / 1e9 # Conversion nanosecondes -> secondes
        tps = eval_tokens / eval_duration_sec if eval_duration_sec > 0 else 0

        print(f"\nPrompt: {p}")
        print(f"• Tokens d'entrée (Prompt) : {prompt_tokens}")
        print(f"• Tokens de sortie (Réponse): {eval_tokens}")
        print(f"• Tokens TOTAUX consommés  : {prompt_tokens + eval_tokens}")
        print(f"• Vitesse de génération   : {tps:.2f} tokens/sec")
        print(f"• Temps total             : {eval_duration_sec:.2f} s")
        print("-" * 50)
