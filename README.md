# PIR

Les fichiers spécifiques de cette branche :
- similarity_over_tasks.py sert à l'étude de la similarité avec les tasks (en reprenant quelques fonctions de similarity_over_datasets.py)
- visualisation.py sert à générer les images, particulièrement la fonction heatmap pour la partie sur les tasks

Les datas utilisés dans cette partie sont stockés dans les dossier d1 et d5 correspondant respectivement aux données des tasks générées avec les dataset 1 et 5.
Les fichiers jsonl sont générés par le code de Oannis, à partir d'eux on génère les fichiers csv de comparaison et les heatmap à partir de ces csv.

└── data
    ├── sim_dataset-prompt
    │   ├── dataset0-similarityFunc0.csv
    │   ├── dataset0-similarityFunc0.csv-hist-repartition.png
    │   ├── dataset0-similarityFunc0.csv-prompt4datasethist-repartition.png
    │   ├── dataset0-similarityFunc0.heatmap.png
    │   ├── dataset0-similarityFunc1.csv
    │   ├── dataset0-similarityFunc1.csv-hist-repartition.png
    │   ├── dataset0-similarityFunc1.heatmap.png
    │   ├── dataset0-similarityFunc2.csv
    │   ├── dataset0-similarityFunc2.csv-hist-repartition.png
    │   ├── dataset0-similarityFunc2.heatmap.png
    │   ├── datasetdetails.jsonl
    │   ├── datasetdetails_cleaned.jsonl
    │   ├── heatmap.png
    │   ├── jaccard_without_embeddings.csv
    │   └── prompts.json
    └── sim_tasks
        ├── 1
        │   ├── dataset1-top_k1-top_p0.75-temp0.3.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 10
        │   ├── dataset10-top_k3-top_p0.75-temp0.9.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 100
        │   ├── settings.txt
        │   ├── similarityFunc0.csv
        │   ├── similarityFunc1.csv
        │   ├── similarityFunc2.csv
        │   ├── similarityFunc3.csv
        │   └── task_embeddings.jsonl
        ├── 2
        │   ├── dataset2-top_k1-top_p0.9-temp0.5.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 20
        │   ├── dataset20-top_k4-top_p0.9-temp1.0.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 21
        │   ├── dataset21-top_k4-top_p0.85-temp0.7.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 3
        │   ├── dataset3-top_k1-top_p0.3-temp0.5.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 4
        │   ├── dataset0-similarityFunc0.csv
        │   ├── dataset0-similarityFunc1.csv
        │   ├── dataset0-similarityFunc2.csv
        │   ├── dataset4-top_k1-top_p0.75-temp0.9.jsonl
        │   ├── settings.txt
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   ├── sim_over_tasks3.csv
        │   ├── similarityFunc0.csv
        │   ├── similarityFunc1.csv
        │   ├── similarityFunc2.csv
        │   ├── similarityFunc3.csv
        │   └── task_embeddings.jsonl
        ├── 5
        │   ├── dataset5-top_k1-top_p0.9-temp0.9.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 6
        │   ├── dataset6-top_k2-top_p0.3-temp0.3.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 7
        │   ├── dataset7-top_k10-top_p0.5-temp0.5.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── 9
        │   ├── dataset9-top_k2-top_p0.3-temp0.5.jsonl
        │   ├── sim_over_tasks0.csv
        │   ├── sim_over_tasks1.csv
        │   ├── sim_over_tasks2.csv
        │   └── sim_over_tasks3.csv
        ├── similarity_over_tasks
        │   ├── d1
        │   │   ├── clean_similarity_over_tasks.csv
        │   │   ├── clean_similarity_over_tasks.heatmap.png
        │   │   ├── dataset1-top_k2-top_p0.3-temp0.6.jsonl
        │   │   ├── dataset1-top_k2-top_p0.4-temp0.4.jsonl
        │   │   ├── dataset1-top_k2-top_p0.6-temp0.3.jsonl
        │   │   ├── dataset1-top_k4-top_p0.3-temp0.6.jsonl
        │   │   ├── dataset1-top_k4-top_p0.4-temp0.4.jsonl
        │   │   ├── dataset1-top_k4-top_p0.6-temp0.3.jsonl
        │   │   ├── datasets_and_tasks.csv
        │   │   ├── datasets_and_tasks.heatmap.png
        │   │   ├── merged_tasks_d1.jsonl
        │   │   ├── similarity_over_tasks.csv
        │   │   └── similarity_over_tasks.heatmap.png
        │   ├── d5
        │   │   ├── clean_similarity_over_tasks.csv
        │   │   ├── clean_similarity_over_tasks.heatmap.png
        │   │   ├── dataset5-top_k2-top_p0.3-temp0.6.jsonl
        │   │   ├── dataset5-top_k2-top_p0.4-temp0.4.jsonl
        │   │   ├── dataset5-top_k2-top_p0.6-temp0.3.jsonl
        │   │   ├── dataset5-top_k4-top_p0.3-temp0.6.jsonl
        │   │   ├── dataset5-top_k4-top_p0.4-temp0.4.jsonl
        │   │   ├── dataset5-top_k4-top_p0.6-temp0.3.jsonl
        │   │   ├── datasets_and_tasks.csv
        │   │   ├── datasets_and_tasks.heatmap.png
        │   │   ├── merged_tasks_d5.jsonl
        │   │   ├── similarity_over_tasks.csv
        │   │   └── similarity_over_tasks.heatmap.png
        │   ├── similarity_over_tasks.csv
        │   └── similarity_over_tasks.heatmap.png
        └── similarity_over_tasks.zip




