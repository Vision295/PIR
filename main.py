import json
from prompt_converter import PromptConverter
from torch.nn.functional import cosine_similarity
from utils import *
import pandas as pd
from csv_manager import CsvManager



with open("datasetdetails_cleaned.jsonl", "r", encoding="utf-8") as file:
    data = [json.loads(line) for line in file]

def get_dataset_description(data:dict, descriptionType:str) -> list[list[str]]:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      datasetDescriptions = []
      for index, entry in enumerate(data):
            if entry[descriptionType]:
                  datasetDescriptions.append([" ".join(entry["task_categories"])])
      return datasetDescriptions

def compute_similarity_over_dataset(data:dict, descriptionType:str, similarityFunction) -> csv_writer:
      """
      inputs : 
      datasetDescriptions : list of list of strings, where each list is a dataset description
      [["desc1"], ["desc2"], ..., ["descN"]]
      similarityFunction : function to compute the similarity between two dataset descriptions
      lambda x : computation(x[0], x[1])
      
      returns : csv_writer
      """
      csvManager = CsvManager("cosine_similarity_matrix.csv")
      datasetDescriptions = get_dataset_description(data, descriptionType)

      size = len(datasetDescriptions)
      # Initialize the similarity dictionary
      similarityDict = [{v[0]: [0 for _ in range(size)]} if i != 0 else {"description": [j[0] for j in datasetDescriptions]} for i, v in enumerate([" "] + datasetDescriptions)]
      for i, v in enumerate([0] + datasetDescriptions):
            for j, w in enumerate(datasetDescriptions):
                  # i == 0 is the description in w, w = {"description" : ["desc_1", "desc_2", ...]}
                  if i != 0:
                        promptConverter = PromptConverter(v, w)
                        promptConverter.generate_embeddings()
                        promptConverter.compute_similarity(similarityFunction)
                        # stores the similarity value in : {"desc_i": [X, X, X, ..., Y, X, X, ...]} 
                        # changes Y where Y is the jth element and desc_i is the ith description
                        similarityDict[i][v[0]][j] = promptConverter.similarities[0].item()
                        
      csvManager.write_similarity_dict_to_csv(similarityDict, "simdir.csv")
      return similarityDict



similarityDict = compute_similarity_over_dataset(data[:3],  "task_categories", lambda x: cosine_similarity(x[0], x[1], dim=1))

print(similarityDict)