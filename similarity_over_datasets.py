import json


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


def get_task_description(fileName:str, descriptionType:str) -> list:
      """
      from data get : [["desc_1"], ["desc_2"], ...]
      where desc_X is the description of the dataset
      """
      with open(fileName, "r", encoding="utf-8") as file:
            return [json.loads(line)[descriptionType] for line in file]
