import pandas as pd
import os
from utils import csv_writer


class FileManager():
      """
      A class to read a CSV file and convert it into a DataFrame.
      """
      
      def __init__(self, *file_paths: str):
            """
            Initialize the CsvReader with the path to the CSV file.
      
            Args:
                  file_path (str): The path to the CSV file.
            """
            if file_paths:
                  self.file_paths = list(file_paths)
                  self.datas = [pd.read_csv(file_path) for file_path in self.file_paths]
      
      def write_similarity_dict_to_csv(self, similarityDict: csv_writer, fileName: str, location:str="data/sim_dataset-prompt"):
            rows = []
            for entry in similarityDict[1:]:
                  for key, values in entry.items():
                        rows.append([key] + values)

            df = pd.DataFrame(rows)

            desc = similarityDict[0]
            df.columns = [*desc.keys()] + list(*desc.values())

            # Check if location is a valid existing directory
            if os.path.isdir(location):
                  output_path = os.path.join(location, fileName)
            elif os.path.isfile(location):
                  output_path = location
            else:
                  print(f"Warning: '{location}' is not a valid directory. Saving file locally.")
                  output_path = fileName  # Save in current working directory
            # Save the DataFrame
            df.to_csv(output_path, index=False)