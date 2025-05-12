import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from utils import repartitionThresholds

class Visualization:
      def __init__(self, file_path, ascending=True):
            self.file_path = file_path
            self.df = pd.DataFrame()
            self.ascending = ascending

      def load_data(self):
            self.df = pd.read_csv(self.file_path, index_col=0)
            # print(self.df.head())

      def top_values(self, column_name, n=5):
            if self.df is not None:
                  top = self.df.sort_values('text-classification', ascending=self.ascending).head(n)
                  print(self.file_path)
                  print(top)
            else:
                  print("Data not loaded yet.")

      def wrap_labels(self, labels, width):
            return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

      def heat_map(self):
            if self.df is not None:
                  numeric_df = self.df.select_dtypes(include='number')

                  if numeric_df.empty:
                        print("Aucune colonne numérique à afficher.")
                        return

                  if self.file_path == 'data/data/sim_dataset-prompt/dataset0-similarityFunc0.csv':
                        numeric_df = numeric_df.round(0).astype(int)

                  n_rows, n_cols = numeric_df.shape
                  cell_size = 1.0 
                  figsize_x = max(10, n_cols * cell_size)
                  figsize_y = max(10, n_rows * cell_size)

                  plt.figure(figsize=(figsize_x, figsize_y))
                  if self.file_path == 'data/data/sim_dataset-prompt/dataset0-similarityFunc0.csv':
                        sns.heatmap(numeric_df, annot=True, cmap='viridis', fmt="d", annot_kws={"size": 8})
                  else:
                        sns.heatmap(numeric_df, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 8})

                  plt.title("Heatmap", fontsize=16)
                  plt.tight_layout()

                  save_path = self.file_path[:-3] + 'heatmap.png'
                  plt.savefig(save_path, dpi=300)
                  plt.close()
                  print(f"heat save to {save_path}")
            else:
                  print("Pas de donnees")


      def zbar_chart_threshold(
                  self, 
                  data:str="text-classification text2text-generation text-generation",
                  prompt4dataset:bool=False,
            ):

            if prompt4dataset:
                  """given a prompt we plot the different thresholds for the dataset"""
                  values = self.df.loc[data].to_dict()
            else:
                  """given a dataset we plot different thresholds for the prompts"""
                  # Convert the DataFrame to a dictionary with descriptions as keys and scores as values
                  values = self.df[data].to_dict()

            repartitionThreshold = repartitionThresholds[int(self.file_path[-5])]
            self.promptsPerThreshold = {}
            for i in repartitionThreshold:
                  self.promptsPerThreshold[i] = list(dict(filter(
                              lambda item: item[1] <= i,
                              values.items()
                  )).keys())
            
            for i in self.promptsPerThreshold:
                  print(i)
                  

      def bar_chart_threshold(self, value):
            if self.df is not None:
                  numeric_df = self.df.select_dtypes(include='number')
                  # seuil
                  filtered_df = numeric_df[numeric_df > value].dropna(axis=1, how='all')

                  # Tracer le graphique à barres
                  plt.figure(figsize=(20, 40))
                  filtered_df.plot(kind='bar', figsize=(20, 40))
                  plt.title(f"Bar chart threshold {value}")
                  plt.ylabel('Values')
                  plt.xlabel('Colonnes')
                  plt.tight_layout()
                  plt.savefig('barchart.png')
                  print("barchart saved to barchart.png")
            else:
                  print("Données non chargées.")
            
      def get_repartition(self, otherData:dict | None=None, nbins:int=90, simDistIndex=0, name:str=""):
            if otherData is not None:
                  all_values = otherData.values()
            else:
                  all_values = []
                  for index, row in self.df.iterrows():
                        for value in row:
                              if type(value) is float:
                                    all_values.append(value)


            plt.figure(figsize=(10, 6))
            plt.hist(all_values, bins=nbins, color='skyblue', edgecolor='black')

            # Add labels and title
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of values of distance using {simDistIndex} on {self.file_path}')

            # Display the plot
            plt.tight_layout()
            plt.savefig(f'{self.file_path}-{name}hist-repartition.png', dpi=300, bbox_inches='tight', transparent=True)


vizualizer = [
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc0.csv', ascending=True),
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc1.csv', ascending=True),
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc2.csv', ascending=False)
]
# for viz in vizualizer:
#       viz.load_data()
#       viz.top_values('text-classification')

# viz = Visualization('data/csv2/dataset0-similarityFunc0-prompt0.csv', ascending=True)
# viz.load_data()
# viz.hit_map()
# viz.bar_chart_threshold(6.5)

print("ok")


#       viz.heat_map()


for i, viz in enumerate(vizualizer):
      viz.load_data()
      viz.heat_map()
      # viz.get_repartition(simDistIndex=i)
      # viz.top_values('text-classification', n=5)

# vizualizer[0].load_data()
# data=" ".join(["text categorization", "document classification", "content labeling", "topic identification"]),
# vizualizer[0].zbar_chart_threshold(
#       data=data,
#       prompt4dataset=True
# )
# vizualizer[0].get_repartition(vizualizer[0].df.loc[data].to_dict(), nbins=10, simDistIndex=0, name="prompt4dataset")
