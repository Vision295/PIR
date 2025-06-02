# PIR

Les fichiers spécifiques de cette branche :
- similarity_over_tasks.py sert à l'étude de la similarité avec les tasks (en reprenant quelques fonctions de similarity_over_datasets.py)
- visualisation.py sert à générer les images, particulièrement la fonction heatmap pour la partie sur les tasks

Les datas utilisés dans cette partie sont stockés dans les dossier d1 et d5 correspondant respectivement aux données des tasks générées avec les dataset 1 et 5.
Les fichiers jsonl sont générés par le code de Oannis, à partir d'eux on génère les fichiers csv de comparaison et les heatmap à partir de ces csv.
data
├── csv2
├── sim_dataset-prompt
└── sim_tasks
    ├── ...
    └── similarity_over_tasks
        ├── d1
        └── d5




