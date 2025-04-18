import pandas as pd
import os
from utils import *


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
      
      def write_similarity_dict_to_csv(self, similarityDict: csv_writer, fileName: str):
            rows = []
            for entry in similarityDict[1:]:
                  for key, values in entry.items():
                        rows.append([key] + values)

            df = pd.DataFrame(rows)

            desc = similarityDict[0]
            df.columns = [*desc.keys()] + list(*desc.values())

            # Write the DataFrame to a CSV file
            os.makedirs("csv", exist_ok=True)
            output_path = os.path.join("csv", fileName)
            df.to_csv(output_path, index=False)