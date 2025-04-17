import json
import torch
from prompt_converter import PromptConverter
from torch.nn.functional import cosine_similarity
from utils import *
import pandas as pd
from csv_manager import CsvManager
from math import pow


def get_dataset_description(datasetName:dict, descriptionType:str) -> list[list[str]]:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      with open(datasetName, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
      
      datasetDescriptions = []
      for index, entry in enumerate(data):
            if entry[descriptionType]:
                  datasetDescriptions.append([" ".join(entry[descriptionType])])
      return datasetDescriptions

def compute_similarity_over_dataset(datasetName:str, descriptionType:str, outputFileName:str, similarityFunction) -> csv_writer:
      """
      inputs : 
      datasetDescriptions : list of list of strings, where each list is a dataset description
      [["desc1"], ["desc2"], ..., ["descN"]]
      similarityFunction : function to compute the similarity between two dataset descriptions
      lambda x : computation(x[0], x[1])
      
      returns : csv_writer
      """
      csvManager = CsvManager()
      datasetDescriptions = get_dataset_description(datasetName, descriptionType)

      size = len(datasetDescriptions)
      # Initialize the similarity dictionary
      similarityDict = [{v[0]: [0 for _ in range(size)]} if i != 0 else {descriptionType: [j[0] for j in datasetDescriptions]} for i, v in enumerate([" "] + datasetDescriptions)]
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
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName)
      return similarityDict


def euclidian(x:int, powDist:float=0.5):
      res = 0
      y = x[0].tolist()[0]
      z = x[1].tolist()[0]
      for i in range(len(y)):
            res += pow((z[i] - y[i]), 1/powDist)
      return torch.tensor([pow(res, powDist)])

cosine = lambda x: cosine_similarity(x[0], x[1], dim=1)

similarityDict = compute_similarity_over_dataset(
      datasetName="datasetdetails_cleaned.jsonl",
      descriptionType="task_categories", 
      outputFileName="disteuclidian3.csv",
      similarityFunction=lambda x : euclidian(x, 0.333)
)

print(similarityDict)