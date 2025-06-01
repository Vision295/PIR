import json

from prompt_converter import PromptConverter
from utils import csv_writer, similarityFunctions, file_names
from file_manager import FileManager
from math import fabs
from torch import Tensor


def get_dataset_description(datasetName:str, descriptionType:str) -> list[list[str]]:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      with open(datasetName, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
      
      datasetDescriptions = []
      for index, entry in enumerate(data):
            if entry[descriptionType]:
                  if type(entry[descriptionType]) is not str:
                        datasetDescriptions.append([" ".join(entry[descriptionType])])
                  else:
                        datasetDescriptions.append([entry[descriptionType]])
      return datasetDescriptions

def get_prompt_description(promptName:str) -> list[list[str]] :
      with open(promptName, "r", encoding="utf-8") as file:
            data = json.load(file)
      
      new_data = []
      
      if type(data) is not str:
            for v in data:
                  new_data.append([" ".join(v)])
      return new_data

def get_task_description(fileName:str, descriptionType:str=None) -> list:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      with open(fileName, "r", encoding="utf-8") as file:
            return [json.loads(line)[descriptionType if descriptionType is not None else f"avg_task_{i}_embedding"] for i, line in enumerate(file)]

def compute_similarity_over_dataset(
            dataset1:list[list[str]], 
            dataset2:list[list[str]], 
            outputFileName:str, 
            similarityFunction, 
            location:str="csv",
            d1Embedding:list[float] | None=None,
            d2Embedding:list[float] | None=None,
      ) -> csv_writer:
      """
      used to compute all the *.csv
      inputs : 
      dataset1 and 2 : lists of list of strings, where each list is a dataset description
      [["desc1"], ["desc2"], ..., ["descN"]]
      OR 
      d1Embedding and d2Embedding : lists of embeddings corresponding to the dataset descriptions

      This function either computes the similarity between the dataset descriptions or uses
      the provided embeddings to compute the similarity.

      similarityFunction : function to compute the similarity between two dataset descriptions
      lambda x : computation(x[0], x[1])
      
      returns : csv_writer
      """

      ### TODO : seperate header from data 
      csvManager = FileManager()

      size1 = len(dataset1)
      # Initialize the similarity dictionary
      similarityDict = [{v[0]: [0.0 for _ in range(size1)]} if i != 0 else {"description": [j[0] for j in dataset1]} for i, v in enumerate([""] + dataset2)]
      for i, v in enumerate([[""]] + dataset2):
            print("step :", i, "out of :", len(dataset2), "steps")
            for j, w in enumerate(dataset1):
                  # i == 0 is the description in w, w = {"description" : ["desc_1", "desc_2", ...]}
                  if i != 0:
                        if d1Embedding is None and d2Embedding is None:
                              promptConverter = PromptConverter(v, w)
                              promptConverter.generate_embeddings()
                        elif d1Embedding is None and d2Embedding is not None:
                              promptConverter = PromptConverter(v)
                              promptConverter.generate_embeddings()
                              promptConverter.embeddings.append(d2Embedding[j])
                        elif d1Embedding is not None and d2Embedding is None:
                              promptConverter = PromptConverter(w)
                              promptConverter.generate_embeddings()
                              promptConverter.embeddings.append(d1Embedding[i - 1])
                        elif d1Embedding is not None and d2Embedding is not None:
                              promptConverter = PromptConverter("kfdjsqmlfjdsklm", "ok1")
                              promptConverter.embeddings = [d1Embedding[i - 1], d2Embedding[j]]
                        # print(promptConverter.embeddings, sum([float(d) for d in promptConverter.embeddings[0][0]]))
                        # exit()
                        promptConverter.compute_similarity(similarityFunction)
                        # stores the similarity value in : {"desc_i": [X, X, X, ..., Y, X, X, ...]} 
                        # changes Y where Y is the jth element and desc_i is the ith description
                        similarity:float = fabs(promptConverter.similarities[0].item())
                        similarityDict[i][v[0]][j] = similarity # type: ignore
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName, location)
      return similarityDict

""" datasetList = [["task_categorization", "task_classification"], [...]]
      OR 
      datasetList = [["one very long task"], ["another long task"], ...]
"""
datasetList = list[list[str]]
def compute_all_distances(
            dataset1:datasetList,
            dataset2:datasetList,
            similarityFunctions:list,
            outputLocation:str
      ) -> None:
      
      """loops through the list of similarity functions and computes the similarity between the datasets"""
      for j, similarityFunction in enumerate(similarityFunctions):
                  print(f"STEP {j} out of {len(similarityFunctions)}")
                  compute_similarity_over_dataset(
                        dataset1=dataset1,
                        dataset2=dataset2,
                        outputFileName=f"similarityFunc{j}.csv",
                        similarityFunction=lambda x: similarityFunction(x),
                        location=outputLocation
                  )
                  print(f"printed in : similarityFunc{j}.csv")


datasets1 = get_dataset_description("data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories") # + \ get_dataset_description of another one
datasets2 = get_prompt_description("data/sim_dataset-prompt/prompts.json")

def compute_sim_task(a:int):
      """used to compute all the data/diff_tasks/*/*/*.csv"""
      for i in file_names[a]:
            d1 = get_task_description(f"data/sim_tasks/diff_duplicates/dataset1/{i}", "task")
            e1 = get_task_description(f"data/sim_tasks/diff_duplicates/dataset1/{i}")
            e2 = get_task_description(f"data/sim_tasks/diff_duplicates/dataset1/{i}", "dataset_embedding")
            compute_similarity_over_dataset(
                  dataset1=[[d] for d in d1],
                  dataset2=[[d1[0]]],
                  outputFileName=f"sim_over_tasks{i}.csv",
                  similarityFunction=similarityFunctions[2],
                  location="data/sim_tasks/diff_duplicates/dataset1",
                  d2Embedding=[[Tensor(e)] for e in e1] if e1 else None, 
                  d1Embedding=[[Tensor(e)] for e in e2] if e2 else None,
            )


compute_sim_task(8)