import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model in 4-bit...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

print("Model loaded.")
print("Device:", next(model.parameters()).device)

prompt = "What is the capital of France? Explain briefly."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

generated_ids = outputs.sequences[0]

input_length = inputs["input_ids"].shape[1]
new_tokens = generated_ids[input_length:]

print("\nGenerated text:")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))

print("\nNumber of generated tokens:", len(new_tokens))
print("Number of score tensors:", len(outputs.scores))

if outputs.scores:
    print("First score tensor shape:", outputs.scores[0].shape)
