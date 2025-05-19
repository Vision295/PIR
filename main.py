import json
import csv
import torch
from prompt_converter import PromptConverter
from torch.nn.functional import cosine_similarity
from torch.linalg import norm
from scipy.stats import wasserstein_distance

def dot_product_similarity(x, y):
    return torch.sum(x * y, dim=1)

# with open("datasetdetails_cleaned.jsonl", "r", encoding="utf-8") as file:
#     data = [json.loads(line) for line in file]


def getDatasetDescriptions(data:dict) -> list[dict]:
      datasetDescriptions = []
      for index, entry in enumerate(data):
            if entry["task_categories"]:
                  datasetDescriptions.append([" ".join(entry["task_categories"])])
      return datasetDescriptions

def computeSimilarityOverDataset(datasetDescriptions:list[dict], similarityFunction) -> list[list[float]]:
      size = len(datasetDescriptions)
      similaritieList = [["" for _ in range(size)] for _ in range(size)]
      for i, v in enumerate(datasetDescriptions):
            for j, w in enumerate(datasetDescriptions):
                  print(i, j)
                  promptConverter = PromptConverter(v, w)
                  promptConverter.generate_embeddings()
                  promptConverter.compute_similarity(similarity_function=similarityFunction)
                  similaritieList[i][j] = promptConverter.similarities[0].item()
                        
      return similaritieList

def writeMatrixOnCSV(matrix:list[list[float]], fileName:str):
      with open(fileName, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(matrix)

def find_high_values(csv_file, threshold=0.98):
      high_value_positions = []

      with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row_idx, row in enumerate(reader):
                  for col_idx, value in enumerate(row[1:], start=1):  # skip description if needed
                        try:
                              if float(value) >= threshold:
                                    if row_idx != col_idx:
                                          high_value_positions.append((row_idx, col_idx, float(value)))
                        except ValueError:
                              pass  # skip non-float entries
      
      index = []

      for tuple in high_value_positions:
            if tuple[0] not in index:
                  index.append(tuple[0])

      return index

def clean_tasks_csv(old_csv, index):
      path = "data/data/sim_tasks/clean_similarity.csv"
      clean_line = []
      with open(old_csv, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row_idx, row in enumerate(reader):
                  if row_idx not in index:
                        clean_line.append(row)
      
      with open(path, "w", newline=",", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(clean_line)

            print(f"enregisté sucessfully dans {path}")








# datasetDescriptions = getDatasetDescriptions(data)
# # similarityList = computeSimilarityOverDataset(datasetDescriptions, lambda x: 1- cosine_similarity(x[0], x[1], dim=1))
# # writeMatrixOnCSV(similarityList, "cosine_similarity_matrix.csv")
# # similarityList = computeSimilarityOverDataset(datasetDescriptions, lambda x: norm(x[0] - x[1], dim=1)) #  Inverser pour que la similarité soit plus grande pour des vecteurs proches 
# # writeMatrixOnCSV(similarityList, "euclidean_similarity_matrix.csv")
# similarityList = computeSimilarityOverDataset(datasetDescriptions, lambda x: wasserstein_distance(x[0], x[1]))
# writeMatrixOnCSV(similarityList, "wasserstein_similarity_matrix.csv")

index = find_high_values("data/data/sim_tasks/similarity_over_tasks.csv")
print(index)
clean_tasks_csv("data/data/sim_tasks/similarity_over_tasks.csv", index)