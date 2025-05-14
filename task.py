import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModel

print("📦 Chargement du modèle Hugging Face...")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
print("✅ Modèle chargé.")


# Liste des fichiers à fusionner
jsonl_files = [
    "./resultats_bis/dataset5-top_k2-top_p0.5-temp0.9.jsonl",
    "./resultats_bis/dataset5-top_k2-top_p0.5-temp0.5.jsonl",
    "./resultats_bis/dataset5-top_k2-top_p0.5-temp0.2.jsonl",
    "./resultats_bis/dataset2-top_k3-top_p0.9-temp0.5.jsonl",
    "./resultats_bis/dataset2-top_k3-top_p0.5-temp0.5.jsonl",
    "./resultats_bis/dataset2-top_k3-top_p0.2-temp0.5.jsonl",
    "./resultats_bis/dataset1-top_k3-top_p0.5-temp0.5.jsonl",
    "./resultats_bis/dataset1-top_k2-top_p0.5-temp0.5.jsonl",
    "./resultats_bis/dataset1-top_k1-top_p0.5-temp0.5.jsonl",
]

# Fichier de sortie fusionné
output_file = "merged_dataset.jsonl"

# Fusion des contenus
merged_data = []

for file_name in jsonl_files:
    file_path = Path(file_name)
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                merged_data.append(json.loads(line))
    else:
        print(f"[!] Fichier non trouvé : {file_name}")

# Sauvegarde du fichier fusionné
with open(output_file, 'w', encoding='utf-8') as out_f:
    for item in merged_data:
        out_f.write(json.dumps(item) + '\n')

print(f"✅ Fusion terminée : {len(merged_data)} tâches écrites dans {output_file}")

# === Étape 2 : Calcul de la similarité entre les tâches ===
import pandas as pd
from tqdm import tqdm
from math import fabs
from prompt_converter import PromptConverter as BasePromptConverter

class PromptConverter(BasePromptConverter):
    def __init__(self, *args, tokenizer=None, model=None):
        super().__init__(*args)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        if model is not None:
            self.model = model

from utils import similarityFunctions

# Charger les descriptions depuis le fichier JSONL fusionné
with open(output_file, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

# Extraire les descriptions à comparer (en s'assurant que "task_categories" existe)
import re

descriptions = []

for entry in dataset:
    task_text = entry.get("task", "")
    match = re.search(r"Task \d+\.\s*(.+)", task_text)
    if match:
        description = match.group(1).strip()
        descriptions.append(description)
    else:
        print(f"[!] Tâche invalide ignorée : {task_text}")


# Choisir la fonction de similarité (cosine_similarity ici)
similarity_func = similarityFunctions[3]

# Comparer toutes les paires de descriptions (sauf identiques)
results = []
for i in tqdm(range(len(descriptions))):
    for j in range(i + 1, len(descriptions)):
        d1 = descriptions[i]
        d2 = descriptions[j]
        converter = PromptConverter(d1, d2, tokenizer=tokenizer, model=model)
        converter.generate_embeddings()
        converter.compute_similarity(similarity_func)
        similarity = fabs(converter.similarities[0].item())
        results.append({
            "description_1": d1,
            "description_2": d2,
            "similarity": similarity
        })

# Enregistrer les résultats dans un fichier CSV
df = pd.DataFrame(results)
df.to_csv("task_similarity_cosine.csv", index=False)
print("✅ Similarité enregistrée dans task_similarity_cosine.csv")

# === Étape finale : transformer les similarités en matrice CSV ===
import pandas as pd

# Charger le CSV contenant les paires
df = pd.read_csv("task_similarity_cosine.csv")

# Créer une matrice de similarité (pivot table)
pivot_df = df.pivot(index="description_1", columns="description_2", values="similarity")

# Supprimer la diagonale (tâche comparée à elle-même)
for task in pivot_df.index:
    if task in pivot_df.columns:
        pivot_df.at[task, task] = ""

# Sauvegarder la matrice dans un nouveau fichier
pivot_df.to_csv("task_similarity_matrix.csv")
print("✅ Matrice enregistrée dans task_similarity_matrix.csv")

