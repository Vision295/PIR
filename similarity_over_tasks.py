import csv
import json
from utils import csv_writer, similarityFunctions
from similarity_over_datasets import get_task_description, get_dataset_description
from prompt_converter import PromptConverter
from utils import csv_writer, similarityFunctions
from file_manager import FileManager
from math import fabs

def compute_similarity_over_tasks(
            data1:list[list[str]], 
            data2:list[list[str]], 
            outputFileName:str, 
            similarityFunction, 
            location:str="csv"
      ) -> csv_writer:
      """
      Computes the similarity between two lists of task descriptions and writes the results to a CSV file.

      This function is very similar to `compute_similarity_over_dataset()` but adapted to compare *tasks*.

      Args:
            data1 : List of task descriptions to compare from.
            data2 : List of task descriptions to compare to.
            similarityFunction : Function used to compute similarity (cosine in this case for the tasks).

      Returns:
            csv_writer: A CSV data structure containing similarity values.
      """   
      csvManager = FileManager()
      size1 = len(data1)

      # Flatten inner lists if needed
      data1 = [item[0] if isinstance(item, list) else item for item in data1]
      size1 = len(data1)
      
      # Initialize the similarity dictionary
      similarityDict = [{v: [0.0 for _ in range(size1)]} if i != 0 else {"description": [j for j in data1]} 
                        for i, v in enumerate([""] + data2)]
      for i, v in enumerate([[""]] + data2):
            print("step :", i, "out of :", len(data2), "steps")
            for j, w in enumerate(data1):
                  # i == 0 is the description in w, w = {"description" : ["desc_1", "desc_2", ...]}
                  if i != 0:
                        promptConverter = PromptConverter(v, w)
                        promptConverter.generate_embeddings()
                        promptConverter.compute_similarity(similarityFunction)                        
                        # Get absolute similarity value
                        similarity:float = fabs(promptConverter.similarities[0].item())
                        similarityDict[i][v][j] = similarity # type: ignore
                        
      print(similarityDict)
      csvManager.write_similarity_dict_to_csv(similarityDict, outputFileName, location)
      return similarityDict


def find_high_values(csv_file, threshold=0.85):
      """
      Finds indices of tasks in a CSV file that are too similar (with score above a threshold).

      Args:
            csv_file : Path to the CSV file with similarity scores.
            threshold : Threshold above which values are considered "too similar".

      Returns:
            list of int: List of row indices to remove due to high similarity.
      """
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
      """
      Removes rows and columns corresponding to high similarity values from the CSV.

      Args:
            old_csv : Path to the original CSV file.
            new_path : Path to save the cleaned CSV file.
            index : List of row/column indices to remove.
      """
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
# Example usage below:

### To generate similarity_over_tasks.csv file :
      # folders = ["d1","d5"]
      # for d in folders:
            # tasks_file = f"data/data/sim_tasks/similarity_over_tasks/{d}/merged_tasks_{d}.jsonl"
            # output_folder = f"data/data/sim_tasks/similarity_over_tasks/{d}/"
            # tasks = get_task_description(tasks_file, "task")
            # compute_similarity_over_tasks(data1=tasks,
            #                               data2=tasks,
            #                               outputFileName="similarity_over_tasks.csv",
            #                               similarityFunction=similarityFunctions[2],
            #                               location=output_folder)

### To generate clean_similarity_over_tasks.csv file : 
      # folders = ["d1","d5"]
      # for d in folders:
            # index = find_high_values(f"data/data/sim_tasks/similarity_over_tasks/{d}/similarity_over_tasks.csv")
            # clean_tasks_csv(f"data/data/sim_tasks/similarity_over_tasks/{d}/similarity_over_tasks.csv",
            #                   f"data/data/sim_tasks/similarity_over_tasks/{d}/clean_similarity_over_tasks.csv",
            #                   index)

### To compare the tasks generated for dataset1 with all the datasets
      index = find_high_values(f"data/data/sim_tasks/similarity_over_tasks/d1/similarity_over_tasks.csv")
      tasks = get_task_description("data/data/sim_tasks/similarity_over_tasks/d1/merged_tasks_d1.jsonl", "task")
      
      # Sort the indices in descending order
      for i in sorted(index, reverse=True):
            if 0 <= i - 1 < len(tasks):
                  tasks.pop(i - 1)
            else:
                  print(f"Index {i - 1} is out of range")
      
      data = get_dataset_description("data/data/sim_dataset-prompt/datasetdetails_cleaned.jsonl", "task_categories")
      compute_similarity_over_tasks(data1=data,
                                    data2=tasks,
                                    outputFileName="datasets_and_tasks.csv",
                                    similarityFunction=similarityFunctions[2],
                                    location="data/data/sim_tasks/similarity_over_tasks/d1/")