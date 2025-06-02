import csv
import json
from utils import csv_writer, similarityFunctions
from similarity_over_datasets import get_task_description, get_dataset_description
from prompt_converter import PromptConverter
from utils import csv_writer, similarityFunctions
from file_manager import FileManager
from math import fabs

def compute_similarity_over_tasks(
            dataset1:list[list[str]], 
            dataset2:list[list[str]], 
            outputFileName:str, 
            similarityFunction, 
            location:str="csv",
            d1Embedding:list[float] | None=None,
            d2Embedding:list[float] | None=None,
      ) -> csv_writer:
      
      csvManager = FileManager()

      size1 = len(dataset1)
      dataset1 = [item[0] if isinstance(item, list) else item for item in dataset1]
      size1 = len(dataset1)
      # Initialize the similarity dictionary
      similarityDict = [{v: [0.0 for _ in range(size1)]} if i != 0 else {"description": [j for j in dataset1]} for i, v in enumerate([""] + dataset2)]
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
                        similarityDict[i][v][j] = similarity # type: ignore
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName, location)
      return similarityDict


def find_high_values(csv_file, threshold=0.85):
      high_value_positions = []

      with open(csv_file, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row_idx, row in enumerate(reader):
                  for col_idx, value in enumerate(row[1:], start=1):  # skip description if needed
                        try:
                              if float(value) > threshold:
                                    if row_idx != col_idx:
                                          high_value_positions.append((row_idx, col_idx, float(value)))
                        except ValueError:
                              pass  # skip non-float entries
      
      index = []

      for tuple in high_value_positions:
            if tuple[0] not in index:
                  index.append(tuple[0])

      return index

def clean_tasks_csv(old_csv, new_path, index):
      path = new_path
      clean_line = []
      with open(old_csv, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row_idx, row in enumerate(reader):
                  if row_idx not in index:
                        cleaned_row = [value for col_idx, value in enumerate(row) if col_idx not in index]
                        clean_line.append(cleaned_row)
      
      with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(clean_line)

            print(f"enregisté sucessfully dans {path}")


if __name__ == '__main__':
      # folders = ["d1","d5"]
      # for d in folders:

###     Pour generer similarity_over_tasks.csv
            # tasks_file = f"data/data/sim_tasks/similarity_over_tasks/{d}/merged_tasks_{d}.jsonl"
            # output_folder = f"data/data/sim_tasks/similarity_over_tasks/{d}/"
            # tasks = get_task_description(tasks_file, "task")
            # compute_similarity_over_tasks(dataset1=tasks,
            #                               dataset2=tasks,
            #                               outputFileName="similarity_over_tasks.csv",
            #                               similarityFunction=similarityFunctions[2],
            #                               location=output_folder)

###     Pour generer clean_similarity_over_tasks.csv
      index = find_high_values(f"data/data/sim_tasks/similarity_over_tasks/d1/similarity_over_tasks.csv")
      # clean_tasks_csv(f"data/data/sim_tasks/similarity_over_tasks/{d}/similarity_over_tasks.csv",
      #                   f"data/data/sim_tasks/similarity_over_tasks/{d}/clean_similarity_over_tasks.csv",
      #                   index)
      tasks = get_task_description("data/data/sim_tasks/similarity_over_tasks/d1/merged_tasks_d1.jsonl", "task")
      
      # Sort the indices in descending order
      for i in sorted(index, reverse=True):
            if 0 <= i - 1 < len(tasks):
                  tasks.pop(i - 1)
            else:
                  print(f"Index {i - 1} is out of range")
      
      data = get_dataset_description("data/data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories")
      compute_similarity_over_tasks(dataset1=data,
                                    dataset2=tasks,
                                    outputFileName="datasets_and_tasks.csv",
                                    similarityFunction=similarityFunctions[2],
                                    location="data/data/sim_tasks/similarity_over_tasks/d1/")