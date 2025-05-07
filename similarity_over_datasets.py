import json

from prompt_converter import PromptConverter
from utils import csv_writer, similarityFunctions
from file_manager import FileManager
from math import fabs

### TODO : Jaccard Distance

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
                  if type(entry) is not str:
                        datasetDescriptions.append([" ".join(entry[descriptionType])])
      return datasetDescriptions

def get_prompt_description(promptName:str) -> list[list[str]] :
      with open(promptName, "r", encoding="utf-8") as file:
            data = json.load(file)
      
      new_data = []
      
      if type(data) is not str:
            for v in data:
                  new_data.append([" ".join(v)])
      return new_data

def compute_similarity_over_dataset(dataset1:list[list[str]], dataset2:list[list[str]], outputFileName:str, similarityFunction, location:str="csv") -> csv_writer:
      """
      inputs : 
      datasets1 and 2 : lists of list of strings, where each list is a dataset description
      [["desc1"], ["desc2"], ..., ["descN"]]
      similarityFunction : function to compute the similarity between two dataset descriptions
      lambda x : computation(x[0], x[1])
      
      returns : csv_writer
      """
      csvManager = FileManager()

      size1 = len(dataset1)
      # Initialize the similarity dictionary
      similarityDict = [{v[0]: [0 for _ in range(size1)]} if i != 0 else {"description": [j[0] for j in dataset1]} for i, v in enumerate([" "] + dataset2)]
      for i, v in enumerate([""] + dataset2):
            print("step :", i, "out of :", len(dataset2), "steps")
            for j, w in enumerate(dataset1):
                  # i == 0 is the description in w, w = {"description" : ["desc_1", "desc_2", ...]}
                  if i != 0:
                        promptConverter = PromptConverter(v, w)
                        promptConverter.generate_embeddings()
                        promptConverter.compute_similarity(similarityFunction)
                        # stores the similarity value in : {"desc_i": [X, X, X, ..., Y, X, X, ...]} 
                        # changes Y where Y is the jth element and desc_i is the ith description
                        similarityDict[i][v[0]][j] = str(fabs(promptConverter.similarities[0].item())) 
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName, location)
      return similarityDict

datasetList = list[list[list[str]]]
def compute_all_distances(
            datasets1:datasetList,
            datasets2:datasetList,
            simialirtyFunctions:list,
            outputLocation:str
      ) -> None:
      for i, dataset in enumerate(datasets1):
            for j, similarityFunction in enumerate(similarityFunctions):
                  for prompt in datasets2:
                        print(f"STEP {j*i+j} out of {len(datasets1)*len(datasets2)*len(similarityFunctions)}, working on dataset {i}, similarityFunction {j}")
                        compute_similarity_over_dataset(
                              dataset1=dataset,
                              dataset2=prompt,
                              outputFileName=f"dataset{i}-similarityFunc{j}.csv",
                              similarityFunction=lambda x: similarityFunction(x),
                              location=outputLocation
                        )
                        print(f"printed in : dataset{i}-similarityFunc{j}.csv")


print(get_dataset_description("datasetdetails_cleaned.jsonl", "task_categories"), get_prompt_description("prompts.json"))

# compute_all_distances(
#       [get_dataset_description("datasetdetails_cleaned.jsonl", "task_categories")],
#       [get_prompt_description("prompts.json")],
#       similarityFunctions
# )

compute_all_distances(
      [get_dataset_description("task_embeddings.jsonl", "task")],
      [get_dataset_description("task_embeddings.jsonl", "task")],
      similarityFunctions,
      outputLocation="sim_tasks"
)