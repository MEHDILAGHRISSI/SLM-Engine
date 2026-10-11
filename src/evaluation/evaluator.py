import torch
import time
import json
import subprocess
import tempfile
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ---------------------------------------------------------
# 1. Configuration du modèle et chargement des poids
# ---------------------------------------------------------
base_model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
lora_path = "outputs_java_gpu/checkpoint-150"

print("=== [1/4] Chargement de Qwen2.5-Coder + Adaptateur LoRA ===")
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
# 2. Jeu de données d'évaluation (Test Suite)
# ---------------------------------------------------------
EVAL_DATASET = [
    {
        "id": "JAVA-SQLI",
        "lang": "java",
        "type": "VULNERABLE",
        "category": "SQL Injection",
        "code": """public void searchUser(String name) { String query = "SELECT * FROM users WHERE name = '" + name + "'"; statement.executeQuery(query); }"""
    },
    {
        "id": "JAVA-SAFE",
        "lang": "java",
        "type": "SECURE",
        "category": "Code Spring Sain",
        "code": """public User findById(Long id) { return userRepository.findById(id).orElseThrow(); }"""
    },
    {
        "id": "PY-CMD",
        "lang": "python",
        "type": "VULNERABLE",
        "category": "Command Injection",
        "code": """import os\ndef execute_user_cmd(cmd):\n    os.system("ping " + cmd)"""
    },
    {
        "id": "PY-SAFE",
        "lang": "python",
        "type": "SECURE",
        "category": "Fonction Python Saine",
        "code": """def add_numbers(a: int, b: int) -> int:\n    return a + b"""
    }
]

# ---------------------------------------------------------
# 3. Fonctions de vérification de compilation / syntaxe
# ---------------------------------------------------------
def check_python_syntax(code_str: str) -> bool:
    """Vérifie si le code Python généré est valide syntaxiquement."""
    try:
        compile(code_str, "<string>", "exec")
        return True
    except Exception:
        return False

def check_java_syntax(code_str: str) -> bool:
    """Vérifie si le code Java généré compile avec javac (si disponible)."""
    if not code_str.strip():
        return False
    
    # Envelopper dans une classe si ce n'est qu'une méthode
    if "class " not in code_str:
        full_code = f"public class DummyTest {{\n {code_str} \n}}"
    else:
        full_code = code_str

    with tempfile.NamedTemporaryDirectory() as tmpdir:
        java_file = os.path.join(tmpdir, "DummyTest.java")
        with open(java_file, "w") as f:
            f.write(full_code)
        try:
            res = subprocess.run(["javac", java_file], capture_output=True, timeout=5)
            return res.returncode == 0
        except FileNotFoundError:
            # Si javac n'est pas installé, validation de présence des accolades
            return full_code.count("{") == full_code.count("}")

# ---------------------------------------------------------
# 4. Boucle d'évaluation
# ---------------------------------------------------------
print("\n=== [2/4] Démarrage de l'évaluation ===")

correct_classifications = 0
syntax_passes = 0
latencies = []

for idx, test in enumerate(EVAL_DATASET, 1):
    prompt = (
        f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
        f"<|im_start|>user\nAnalysez la sécurité de ce code {test['lang']} :\n```{test['lang']}\n{test['code']}\n```<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    start_t = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, temperature=0.1, do_sample=False)
    elapsed = time.time() - start_t
    latencies.append(elapsed)
    
    response = tokenizer.decode(out[0], skip_special_tokens=False)
    reply = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    
    # Évaluation de la classification
    is_detected_vuln = "Code Vulnérable" in reply
    expected_vuln = (test["type"] == "VULNERABLE")
    classification_correct = (is_detected_vuln == expected_vuln)
    if classification_correct:
        correct_classifications += 1
        
    # Vérification de la syntaxe du code suggéré (si vulnérable)
    syntax_ok = True
    if is_detected_vuln and "```" in reply:
        try:
            extracted_code = reply.split("```")[1].replace(test['lang'], '').strip()
            if test['lang'] == 'python':
                syntax_ok = check_python_syntax(extracted_code)
            elif test['lang'] == 'java':
                syntax_ok = check_java_syntax(extracted_code)
        except IndexError:
            syntax_ok = False
            
    if syntax_ok:
        syntax_passes += 1
        
    print(f"\nTest #{idx} [{test['id']}] - {test['category']}")
    print(f"➜ Diagnostic attendu : {test['type']}")
    print(f"➜ Diagnostic modèle : {'VULNERABLE' if is_detected_vuln else 'SECURE'}")
    print(f"➜ Classification Réussie : {'✅ OUI' if classification_correct else '❌ NON'}")
    print(f"➜ Temps de réponse : {elapsed:.2f}s")

# ---------------------------------------------------------
# 5. Calcul des métriques globales
# ---------------------------------------------------------
total_tests = len(EVAL_DATASET)
accuracy = (correct_classifications / total_tests) * 100
syntax_rate = (syntax_passes / total_tests) * 100
avg_latency = sum(latencies) / total_tests

print("\n" + "="*50)
print("             RÉSULTATS DU BENCHMARK")
print("="*50)
print(f"• Précision de Détection (Accuracy) : {accuracy:.1f}% ({correct_classifications}/{total_tests})")
print(f"• Valideur Syntaxique (Pass Rate)   : {syntax_rate:.1f}%")
print(f"• Latence Moyenne par Tâche         : {avg_latency:.2f} secondes")
print("="*50)
