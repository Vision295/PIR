import json
from similarity import Similarity


with open("datasetdetails.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

count = 0
for entry in data:
    count += 1
    if count == 1:
        print(f"ID: {entry['id']}, Text: {entry['name']}")
        text = str(entry['task_categories'])

similarity = Similarity(text, text[:-1])
similarity.generate_embeddings()