import json
import csv
from prompt_converter import PromptConverter
from torch.nn.functional import cosine_similarity

with open("datasetdetails_cleaned.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

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

datasetDescriptions = getDatasetDescriptions(data)
similarityList = computeSimilarityOverDataset(datasetDescriptions, lambda x: cosine_similarity(x[0], x[1], dim=1))
writeMatrixOnCSV(similarityList, "cosine_similarity_matrix.csv")
