import csv
import json
from utils import csv_writer, similarityFunctions
from similarity_over_datasets import compute_similarity_over_dataset, get_task_description


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
###     Pour generer similarity_over_tasks.csv
#     tasks_file = "data/data/sim_tasks/similarity_over_tasks/d5/merged_tasks_d5.jsonl"
#     tasks = get_task_description(tasks_file, "task")
#     compute_similarity_over_dataset(dataset1=tasks,
#                                     dataset2=tasks,
#                                     outputFileName="similarity_over_tasks.csv",
#                                     similarityFunction=similarityFunctions[2],
#                                     location="data/data/sim_tasks/similarity_over_tasks/d5/")

###     Pour generer clean_similarity_over_tasks.csv
    index = find_high_values("data/data/sim_tasks/similarity_over_tasks/d1/similarity_over_tasks.csv")
    print(index)
    clean_tasks_csv("data/data/sim_tasks/similarity_over_tasks/d1/similarity_over_tasks.csv",
                    "data/data/sim_tasks/similarity_over_tasks/d1/clean_similarity_over_tasks.csv",
                    index)