import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
lora_path = "outputs_java_gpu/checkpoint-150"

print("Chargement du modèle de base et du tokenizer...")
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

# Chargement de l'adaptateur LoRA depuis le checkpoint 150
print("Fusion de l'adaptateur LoRA...")
model = PeftModel.from_pretrained(base_model, lora_path, local_files_only=True)

# Code Java vulnérable à tester
code_a_tester = """
public void getUserData(String userId) {
    String query = "SELECT * FROM users WHERE id = '" + userId + "'";
    Statement statement = connection.createStatement();
    ResultSet resultSet = statement.executeQuery(query);
}
"""

prompt = (
    f"<|im_start|>system\nVous êtes un agent expert en analyse statique de code.<|im_end|>\n"
    f"<|im_start|>user\nAnalysez la sécurité de ce code java :\n```java\n{code_a_tester}\n```<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGénération du diagnostic par l'agent...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False
    )

reponse = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(reponse)
