# PIR

Chaque fichier est le résultats de l'exécution d'un script.
Est commenté et mentionné par """used to ... """ chaque fonction générant des fichiers
Toutes les raws data (datasets et calculs de similarités) sont dans le dossier data/
Toutes les visualisations et graphes pour visualiser les données sont dans le dossier visuals/

LES FICHIERS :

hugging_face_transformer_test.py 
    inchangé

file_manager.py 
    edit et transform des formats dict au format .csv 

prompt_converter.py
    copie de hugging_face_transformer_test.py avec quelques modif pour pouvoir prendre autant d'inputs que voulu 
        (list[str] -> list[tokenized] -> list[embeddings])
        et a la possibilité de faire .compute_similarity(function)

similarity_over_dataset.py 
    prends les datasets et les prompts et applique les différentes mesures de similaritées sur les différents prompts et sur les différents datasets 
        (utilisant tous les auters fichiers)

visualization.py
    permets de générer les fichiers dans visuals/ à partir de data/ (heatmap, repartition, histogrammes, ... )

utils.py
    des paramètres fix stockées ici pour être réutilisés dans chaque fichier

data
├── csv2
│   ├── dataset0-similarityFunc0-prompt0.csv
│   ├── disteuclidian3prompt.csv
│   └── disteuclidian4prompt.csv
├── sim_dataset-prompt
│   ├── dataset0-similarityFunc0.csv
│   ├── dataset0-similarityFunc1.csv
│   ├── dataset0-similarityFunc2.csv
│   ├── datasetdetails_cleaned.jsonl
│   ├── datasetdetails.jsonl
│   ├── jaccard_without_embeddings.csv
│   └── prompts.json
└── sim_tasks
    ├── diff_args
    │   ├── dataset1
    │   │   ├── dataset1-top_k2-top_p0.3-temp0.6.jsonl
    │   │   ├── dataset1-top_k2-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset1-top_k4-top_p0.3-temp0.6.jsonl
    │   │   ├── dataset1-top_k4-top_p0.4-temp0.4.jsonl
    │   │   ├── dataset1-top_k4-top_p0.6-temp0.3.jsonl
    │   │   ├── precision
    │   │   │   ├── dataset1-top_k2-top_p0.3-temp0.6.jsonl
    │   │   │   ├── dataset1-top_k2-top_p0.4-temp0.4.jsonl
    │   │   │   ├── dataset1-top_k2-top_p0.6-temp0.3.jsonl
    │   │   │   ├── dataset1-top_k4-top_p0.3-temp0.6.jsonl
    │   │   │   ├── dataset1-top_k4-top_p0.4-temp0.4.jsonl
    │   │   │   ├── dataset1-top_k4-top_p0.6-temp0.3.jsonl
    │   │   │   ├── dataset7-top_k2-top_p0.5-temp0.2.jsonl
    │   │   │   ├── dataset7-top_k3-top_p0.2-temp0.5.jsonl
    │   │   │   ├── dataset7-top_k3-top_p0.5-temp0.5.jsonl
    │   │   │   ├── sim_over_tasksdataset1-top_k2-top_p0.3-temp0.6.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset1-top_k2-top_p0.4-temp0.4.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset1-top_k2-top_p0.6-temp0.3.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset1-top_k4-top_p0.3-temp0.6.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset1-top_k4-top_p0.4-temp0.4.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset1-top_k4-top_p0.6-temp0.3.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset7-top_k2-top_p0.5-temp0.2.jsonl.csv
    │   │   │   ├── sim_over_tasksdataset7-top_k3-top_p0.2-temp0.5.jsonl.csv
    │   │   │   └── sim_over_tasksdataset7-top_k3-top_p0.5-temp0.5.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-top_k2-top_p0.3-temp0.6.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-top_k2-top_p0.6-temp0.3.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-top_k4-top_p0.3-temp0.6.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-top_k4-top_p0.4-temp0.4.jsonl.csv
    │   │   └── sim_over_tasksdataset1-top_k4-top_p0.6-temp0.3.jsonl.csv
    │   ├── dataset3
    │   └── dataset5
    │       ├── dataset5-top_k2-top_p0.3-temp0.6.jsonl
    │       ├── dataset5-top_k2-top_p0.6-temp0.3.jsonl
    │       ├── dataset5-top_k4-top_p0.3-temp0.6.jsonl
    │       ├── dataset5-top_k4-top_p0.4-temp0.4.jsonl
    │       ├── dataset5-top_k4-top_p0.6-temp0.3.jsonl
    │       ├── sim_over_tasksdataset5-top_k2-top_p0.3-temp0.6.jsonl.csv
    │       ├── sim_over_tasksdataset5-top_k2-top_p0.6-temp0.3.jsonl.csv
    │       ├── sim_over_tasksdataset5-top_k4-top_p0.3-temp0.6.jsonl.csv
    │       ├── sim_over_tasksdataset5-top_k4-top_p0.4-temp0.4.jsonl.csv
    │       └── sim_over_tasksdataset5-top_k4-top_p0.6-temp0.3.jsonl.csv
    ├── diff_datasets
    │   ├── dataset0-top_k1-top_p0.3-temp0.3.jsonl
    │   ├── dataset10-top_k3-top_p0.75-temp0.9.jsonl
    │   ├── dataset11-top_k3-top_p0.9-temp0.9.jsonl
    │   ├── dataset1-top_k1-top_p0.75-temp0.3.jsonl
    │   ├── dataset20-top_k4-top_p0.9-temp1.0.jsonl
    │   ├── dataset21-top_k4-top_p0.85-temp0.7.jsonl
    │   ├── dataset2-top_k1-top_p0.9-temp0.5.jsonl
    │   ├── dataset3-top_k1-top_p0.3-temp0.5.jsonl
    │   ├── dataset4-top_k1-top_p0.75-temp0.9.jsonl
    │   ├── dataset5-top_k1-top_p0.9-temp0.9.jsonl
    │   ├── dataset6-top_k2-top_p0.3-temp0.3.jsonl
    │   ├── dataset7-top_k10-top_p0.5-temp0.5.jsonl
    │   └── dataset9-top_k2-top_p0.3-temp0.5.jsonl
    ├── diff_duplicates
    │   ├── dataset1
    │   │   ├── dataset1-seed200-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset1-seed218-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset1-seed242-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset1-seed254-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── sim_over_tasksdataset1-seed200-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-seed218-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   ├── sim_over_tasksdataset1-seed242-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   └── sim_over_tasksdataset1-seed254-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   ├── dataset3
    │   │   ├── dataset3-seed206-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset3-seed218-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset3-seed242-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── dataset3-seed254-top_k3-top_p0.6-temp0.3.jsonl
    │   │   ├── sim_over_tasksdataset3-seed206-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   ├── sim_over_tasksdataset3-seed218-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   ├── sim_over_tasksdataset3-seed242-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   │   └── sim_over_tasksdataset3-seed254-top_k3-top_p0.6-temp0.3.jsonl.csv
    │   └── dataset5
    │       ├── dataset5-seed206-top_k3-top_p0.6-temp0.3.jsonl
    │       ├── dataset5-seed218-top_k3-top_p0.6-temp0.3.jsonl
    │       ├── dataset5-seed242-top_k3-top_p0.6-temp0.3.jsonl
    │       ├── dataset5-seed254-top_k3-top_p0.6-temp0.3.jsonl
    │       ├── sim_over_tasksdataset5-seed206-top_k3-top_p0.6-temp0.3.jsonl.csv
    │       ├── sim_over_tasksdataset5-seed218-top_k3-top_p0.6-temp0.3.jsonl.csv
    │       ├── sim_over_tasksdataset5-seed242-top_k3-top_p0.6-temp0.3.jsonl.csv
    │       └── sim_over_tasksdataset5-seed254-top_k3-top_p0.6-temp0.3.jsonl.csv
    └── task_selection
        ├── merged_dataset.jsonl
        └── similarity_over_tasks.csv


------------------------------------------------------

visuals
├── sim_dataset-prompt
│   ├── barchart.png
│   ├── dataset0-similarityFunc0.csv-hist-repartition.png
│   ├── dataset0-similarityFunc0.csv-prompt4datasethist-repartition.png
│   ├── dataset0-similarityFunc0.heatmap.png
│   ├── dataset0-similarityFunc1.csv-hist-repartition.png
│   ├── dataset0-similarityFunc1.heatmap.png
│   ├── dataset0-similarityFunc2.csv-hist-repartition.png
│   ├── dataset0-similarityFunc2.heatmap.png
│   ├── jaccard_without_embeddings.csv-hist-repartition.png
│   └── prompts.json
└── sim_tasks
    ├── diff_args
    │   ├── dataset1
    │   │   ├── hist_mean_scores_visual_no_duplicates.png
    │   │   ├── hist_scores_visual_no_duplicates.png
    │   │   ├── hq_hist_scores_visual_no_duplicates.png
    │   │   ├── precision
    │   │   │   ├── hist_mean_scores_visual_no_duplicates.png
    │   │   │   ├── hist_scores_visual_no_duplicates.png
    │   │   │   ├── hq_hist_scores_visual_no_duplicates.png
    │   │   │   └── results.txt
    │   │   └── results.txt
    │   ├── dataset3
    │   └── dataset5
    │       ├── hist_mean_scores_visual_no_duplicates.png
    │       ├── hist_scores_visual_no_duplicates.png
    │       ├── hq_hist_scores_visual_no_duplicates.png
    │       └── results.txt
    ├── diff_duplicates
    │   ├── dataset1
    │   │   ├── hist_mean_scores_visual_no_duplicates.png
    │   │   ├── hist_scores_visual_no_duplicates.png
    │   │   ├── hq_hist_scores_visual_no_duplicates.png
    │   │   └── results.txt
    │   ├── dataset3
    │   │   ├── hist_mean_scores_visual_no_duplicates.png
    │   │   ├── hist_scores_visual_no_duplicates.png
    │   │   ├── hq_hist_scores_visual_no_duplicates.png
    │   │   └── results.txt
    │   └── dataset5
    │       ├── hist_mean_scores_visual_no_duplicates.png
    │       ├── hist_scores_visual_no_duplicates.png
    │       ├── hq_hist_scores_visual_no_duplicates.png
    │       └── results.txt
    └── task_selection
        └── similarity_over_tasks.heatmap.png
    

