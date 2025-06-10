# PIR

Cette branche permet de calculer différentes mesures de similarité (distance de Minkowski, cosine, Jaccard) entre des descriptions de datasets ou de prompts, puis de visualiser les résultats sous forme de fichiers csv, heatmaps et histogrammes.

## Installation

Installer les bibliothèques requises avec la commande suivante : pip install -r requirements.txt

## Structure du projet

- similarity_over_datasets.py : Calcul des similarités et distances, génération des fichiers csv.
- visualization.py : Génération des heatmaps et histogrammes à partir des fichiers csv.
- utils.py : Fonctions utilitaires pour les mesures de similarité.
- data : Contient les fichiers de données et les résultats.

## Exécution du code

- lancer le script similarity_over_datasets.py pour générer les fichiers csv qui seront stockés dans data/sim_dataset-prompt/ et data/sim_tasks/
- lancer le script visualization.py pour créer les heatmaps et les histogrammes à partir des fichiers csv.
