from transformers import AutoTokenizer, AutoModel
import torch
import json

# Installation instructions: https://huggingface.co/docs/transformers/en/installation

# Load a pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Prompt (scraped from https://huggingface.co/docs/hub/en/datasets-cards)
with open("datasetdetails.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

count = 0
for entry in data:
    count += 1
    if count == 1:
        print(f"ID: {entry['id']}, Text: {entry['name']}")
        text = str(entry['task_categories'])

#text = ["Average temperature tomorrow in Lyon."]
#text = ["Temperature in Lyon 2010-2025."]
inputs = tokenizer(text, return_tensors="pt", truncation=False)

print(f"Number of tokens: {inputs["input_ids"].shape[1]}")

print(*inputs)

# Generate embeddings
with torch.no_grad():  # No gradient calculation needed for inference
    outputs = model(**inputs)

# Extract the embeddings (typically use [CLS] token or mean pooling)
embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
print("Embeddings shape:", embeddings.shape)
print("Sample embedding:", embeddings[0][:5])  # First 5 values of the first embedding