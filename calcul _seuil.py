import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Fonctions de chargement des données
# -----------------------------

def load_list_of_lists_from_json(file_path):
    """
    Charge un fichier JSON contenant une liste de listes de chaînes de caractères.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(sublist, list) and all(isinstance(item, str) for item in sublist) for sublist in data):
        raise ValueError("Le fichier JSON doit contenir une liste de listes de chaînes de caractères.")

    return data

def load_list_of_lists_from_json2(file_path):
    """
    Charge un fichier JSON contenant une liste de listes, puis fusionne chaque sous-liste en une chaîne unique.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(sublist, list) and all(isinstance(item, str) for item in sublist) for sublist in data):
        raise ValueError("Le fichier JSON doit contenir une liste de listes de chaînes de caractères.")

    return [' '.join(sublist) for sublist in data]

def load_csv_as_dict(file_path, orient_by='columns'):
    """
    Charge un fichier CSV et le retourne sous forme de dictionnaire orienté par colonnes ou lignes.
    """
    df = pd.read_csv(file_path)

    if orient_by == 'columns':
        return df.to_dict(orient='list')
    elif orient_by == 'rows':
        return df.to_dict(orient='index')
    else:
        raise ValueError("Le paramètre 'orient_by' doit être 'columns' ou 'rows'")

# -----------------------------
# Fonctions de traitement des similarités
# -----------------------------

def remove_duplicates(strings):
    """
    Supprime les doublons d'une liste tout en préservant l'ordre d'apparition.
    """
    seen = set()
    result = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result

def threshold_similarity_for_prompt(data_similarity, threshold, target_prompt, prompt_column="prompt"):
    """
    Affiche les similarités d'un prompt dans différentes tranches de seuils.
    """
    if prompt_column not in data_similarity:
        raise ValueError(f"Colonne absente: {prompt_column}")

    prompts = data_similarity[prompt_column]
    if target_prompt not in prompts:
        raise ValueError(f"Prompt inconnu: {target_prompt}")

    index = prompts.index(target_prompt)
    sim_columns = [col for col in data_similarity if col != prompt_column]

    threshold = sorted(threshold)
    tranches = [(0.0, t) for t in threshold] + [(threshold[-1], 1.0)]
    tranche_results = {tranche: [] for tranche in tranches}

    for sim_col in sim_columns:
        score = data_similarity[sim_col][index]
        for (low, high) in tranches:
            if low < score <= high:
                tranche_results[(low, high)].append((sim_col, score))
                break

    print(f"\nRésultats pour le prompt: {target_prompt}\n")
    for (low, high), items in tranche_results.items():
        print(f"Tranche ({low}, {high}]:")
        for col, score in sorted(items, key=lambda x: x[1]):
            print(f"  - {col}: {score}")
        if not items:
            print("  (aucune)")

def export_below_threshold_for_prompts(data_similarity, threshold, target_prompts, output_file, prompt_column="description"):
    """
    Exporte dans un fichier CSV toutes les similarités inférieures ou égales à un seuil donné.
    """
    if prompt_column not in data_similarity:
        raise ValueError(f"Colonne absente: {prompt_column}")

    prompts = data_similarity[prompt_column]
    sim_columns = [col for col in data_similarity if col != prompt_column]
    results = []

    for target_prompt in target_prompts:
        if target_prompt not in prompts:
            print(f"Prompt ignoré : {target_prompt}")
            continue

        index = prompts.index(target_prompt)

        for sim_col in sim_columns:
            score = data_similarity[sim_col][index]
            if score <= threshold:
                results.append({
                    "prompt": target_prompt,
                    "similarity_column": sim_col,
                    "score": score
                })

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prompt", "similarity_column", "score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Export terminé : {len(results)} entrées enregistrées dans '{output_file}'")

def ROC_ratio(data_similarity, target_tasks, output_file, prompt_column="description", target_prompt="prompt1"):
    """
    Calcule la précision (TP / (TP + FP)) pour divers seuils et exporte les résultats dans un CSV.
    """
    if prompt_column not in data_similarity:
        raise ValueError(f"Colonne absente: {prompt_column}")

    descriptions = data_similarity[prompt_column]
    if target_prompt not in descriptions:
        raise ValueError(f"Prompt cible absent: {target_prompt}")

    prompt_index = descriptions.index(target_prompt)
    task_columns = [col for col in data_similarity if col != prompt_column]

    all_scores = [score for task in task_columns for score in data_similarity[task]]
    global_max = max(all_scores)
    thresholds = np.linspace(0, global_max, 101)
    results = []

    for threshold in thresholds:
        TP, FP = 0, 0

        for task in task_columns:
            score_target = data_similarity[task][prompt_index]
            if task in target_tasks and score_target < threshold:
                TP += 1

            for i, other_prompt in enumerate(descriptions):
                if other_prompt != target_prompt:
                    if data_similarity[task][i] < threshold:
                        FP += 1
                        break

        total = TP + FP
        ratio = TP / total if total > 0 else 0.0
        results.append({"threshold": round(threshold, 4), "TP": TP, "FP": FP, "TP/(TP+FP)": round(ratio, 4)})

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "TP", "FP", "TP/(TP+FP)"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Fichier ROC ratio enregistré : {output_file}")

# -----------------------------
# Fonctions de visualisation
# -----------------------------

def plot_ROC_ratio(csv_file, title="ROC Ratio Curve"):
    """
    Trace le ratio TP / (TP + FP) en fonction du seuil.
    """
    thresholds, ratios = [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(float(row["threshold"]))
            ratios.append(float(row["TP/(TP+FP)"]))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, ratios, marker='o', linestyle='-', color='blue', label="TP / (TP + FP)")
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ROC_details(csv_file, title="Détails de la courbe ROC", output_file="roc_details.png"):
    """
    Trace TP, FP et le ratio TP/(TP+FP) et enregistre le résultat en image.
    """
    thresholds, tp_list, fp_list, ratio_list = [], [], [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(float(row["threshold"]))
            tp_list.append(int(row["TP"]))
            fp_list.append(int(row["FP"]))
            ratio_list.append(float(row["TP/(TP+FP)"]))

    plt.figure(figsize=(12, 6))
    plt.plot(thresholds, tp_list, label="TP", color="green", linestyle='--', marker='o')
    plt.plot(thresholds, fp_list, label="FP", color="red", linestyle='--', marker='x')
    plt.xlabel("Seuil")
    plt.ylabel("Valeurs")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, format="png")
    plt.close()
    print(f"Graphique enregistré : {output_file}")

def plot_tp_fp_ratio(csv_file):
    """
    Trace le ratio TP / FP en fonction du seuil.
    """
    thresholds, tp_fp_ratios = [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            TP = int(row["TP"])
            FP = int(row["FP"])
            ratio = TP / FP if FP > 0 else float('inf')
            thresholds.append(threshold)
            tp_fp_ratios.append(ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, tp_fp_ratios, label="TP / FP", color='red')
    plt.xlabel("Threshold")
    plt.ylabel("TP / FP")
    plt.title("Évolution du ratio TP / FP")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_roc_curve(csv_file, total_positives, total_negatives, output_file="roc_curve.png"):
    """
    Trace une courbe ROC (TPR vs FPR) et sauvegarde le résultat en image.
    """
    thresholds, tpr_list, fpr_list = [], [], []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            threshold = float(row["threshold"])
            TP = int(row["TP"])
            FP = int(row["FP"])
            FN = total_positives - TP
            TN = total_negatives - FP
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            thresholds.append(threshold)
            tpr_list.append(TPR)
            fpr_list.append(FPR)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, label="ROC Curve", color='blue')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Courbe ROC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, format="png")
    plt.close()
    print(f"Courbe ROC enregistrée : {output_file}")
