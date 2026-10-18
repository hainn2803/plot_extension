from datasets import load_dataset
from mcqa_neural_net import load_gemma_model
import torch

MODEL = "google/gemma-2-2b-it"

ds = load_dataset("fengyang0317/prontoqa")
test = ds["test"]


model, tokenizer = load_gemma_model()
model.eval()

correct = 0

for i in range(10):
    ex = test[i]
    prompt = f'Solve the problem and answer True or False. {ex["question"]} Answer:'
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    pred = "True" if "True" in text else "False"
    gold = ex["answer"]
    correct += pred == gold

    print("=" * 80)
    print(ex["question"])
    print("Gold:", gold)
    print("Pred:", pred)
    print("Raw:", text)

print("Accuracy:", correct / 10)