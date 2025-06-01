import json

from prompt_converter import PromptConverter
from utils import csv_writer, similarityFunctions
from file_manager import FileManager
from math import fabs
from torch import Tensor

def list_to_set(list1:list[list[str]]) -> set:
      """
      converti une liste de listes en un set
      """
      return set(item[0] for item in list1)

def jaccard_index(set1:set, set2:set) -> float:
      """
      Jaccard index entre 2 sets : |A n B| / |A u B|
      """
      intersection = len(set1.intersection(set2))
      union = len(set1.union(set2))
      if union != 0:
            return intersection / union
      else:
            return 0.0
      
def compute_jaccard_index_over_dataset(dataset1:list[list[str]], dataset2:list[list[str]]) -> csv_writer:
      """
      applique l'index de jaccard directement sur les datasets
      """
      csvManager = FileManager()
      size1 = len(dataset1)
      # Initialize the similarity dictionary
      similarityDict = [{v[0]: [0 for _ in range(size1)]} if i != 0 else {"description": [j[0] for j in dataset1]} for i, v in enumerate([" "] + dataset2)]
      for i, v in enumerate([0] + dataset2):
            print("step :", i, "out of :", len(dataset2), "steps")
            for j, w in enumerate(dataset1):
                  # i == 0 is the description in w, w = {"description" : ["desc_1", "desc_2", ...]}
                  if i != 0:
                        similarityDict[i][v[0]][j] = jaccard_index(list_to_set(v), list_to_set(w))
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, "jaccard_without_embeddings.csv")
      return similarityDict

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
      inputs : 
      dataset1 and 2 : lists of list of strings, where each list is a dataset description
      [["desc1"], ["desc2"], ..., ["descN"]]
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
                        else:
                              promptConverter = PromptConverter("ok", "ok")
                              print(type(d2Embedding), type(d2Embedding) is None)
                              promptConverter.embeddings = [d1Embedding[i - 1], d2Embedding[j]]
                        promptConverter.compute_similarity(similarityFunction)
                        # stores the similarity value in : {"desc_i": [X, X, X, ..., Y, X, X, ...]} 
                        # changes Y where Y is the jth element and desc_i is the ith description
                        similarity:float = fabs(promptConverter.similarities[0].item())
                        similarityDict[i][v[0]][j] = similarity # type: ignore
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName, location)
      return similarityDict

"""
      datasetList = [["task_categorization", "task_classification"], [...]]
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

      #jaccard index avec des chaines de caractères sur les datasets directement
      # compute_jaccard_index_over_dataset(
      #       dataset1=dataset1,
      #       dataset2=dataset2
      # )
      
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



def get_task_description(fileName:str, descriptionType:str) -> list:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      with open(fileName, "r", encoding="utf-8") as file:
            return [json.loads(line)[descriptionType] for line in file]



# compute_all_distances(
#       [get_dataset_description("data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories")],
#       [get_prompt_description("data/sim_dataset-prompt/prompts.json")],
#       similarityFunctions
# )

datasets1 = get_dataset_description("data/data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories") # + \ get_dataset_description of another one
datasets2 = get_prompt_description("data/data/sim_dataset-prompt/prompts.json")

file_names = [
      "dataset1-top_k1-top_p0.75-temp0.3.jsonl",
      "dataset2-top_k1-top_p0.9-temp0.5.jsonl",
      "dataset3-top_k1-top_p0.3-temp0.5.jsonl",
      "dataset4-top_k1-top_p0.75-temp0.9.jsonl",
      "dataset5-top_k1-top_p0.9-temp0.9.jsonl",
      "dataset6-top_k2-top_p0.3-temp0.3.jsonl",
      "dataset7-top_k10-top_p0.5-temp0.5.jsonl",
      "dataset9-top_k2-top_p0.3-temp0.5.jsonl",
      "dataset10-top_k3-top_p0.75-temp0.9.jsonl",
      "dataset20-top_k4-top_p0.9-temp1.0.jsonl",
      "dataset21-top_k4-top_p0.85-temp0.7.jsonl"
]



d1 = get_dataset_description(f"data/data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories")
d2 = get_prompt_description(f"data/data/sim_dataset-prompt/prompts.json")

compute_similarity_over_dataset(
      dataset1= d1,
      dataset2= d2 ,
      outputFileName="dataset0-similarityFunc3.csv",
      similarityFunction=similarityFunctions[3],
      location=f"data/data/sim_dataset-prompt",
)
