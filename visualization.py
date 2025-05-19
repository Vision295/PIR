import pandas as pd
from matplotlib import pyplot as plt
from regex import W
import seaborn as sns
import textwrap
from utils import repartitionThresholds, get_name_from_path

class Visualization:
      def __init__(self, file_path, ascending=True):
            self.file_path = file_path
            self.df = pd.DataFrame()
            self.ascending = ascending

      def load_data(self):
            self.df = pd.read_csv(self.file_path, index_col=0)

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
                  if self.file_path == 'data/sim_tasks/task_selection/similarity_voer_tasks.csv':
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
                  plt.savefig('barchart.png', transparent=False)
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
            plt.savefig(f'{self.file_path}-{name}hist-repartition.png', dpi=300, bbox_inches='tight', transparent=False)


      def compute_average_std_score(self, average_score:bool=True):
            if self.df is not None:
                  numeric_df = self.df.select_dtypes(include='number')
                  average_scores = numeric_df.values.mean()
                  std_scores = numeric_df.values.std()
                  return average_scores if average_score else std_scores
            else:
                  print("Data not loaded yet.")
                  return None
      

def get_n_highest_task_values(vizList:list[Visualization]) -> list[tuple]:
      """
      Get the top n values for a specific task across multiple visualizations.
      """
      results = []
      vizDict = {}
      for viz in vizList:
            for i, (k, v) in enumerate(viz.df.to_dict().items()):
                  vizDict[get_name_from_path(viz.file_path) + str(i)] = list(v.values())[0]
      results = sorted(vizDict.items(), key=lambda x: x[1], reverse=True)
      
      return results

def get_best_tasks_args(vizList:list[Visualization], average:bool=True) -> list[tuple]:
      scores = {}
      for viz in vizList:
            viz.load_data()
            scores[get_name_from_path(viz.file_path)] = viz.compute_average_std_score(average)
      sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
      return sorted_scores

def plot_best_values(vizList:list[Visualization], save_path:str, n:int=5):
      """
      Plot the best arguments for a specific task across multiple visualizations.
      """
      for viz in vizList : viz.load_data()
      keys = [get_name_from_path(viz.file_path) for viz in vizList]
      data = []
      for viz in vizList:
            data += [list(viz.df.select_dtypes(include='number').values[0])]
      
      print(data)

      colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22'   # yellow-green
      ]

      plt.figure(figsize=(10, 6))
      plt.hist(
            data,
            bins=2*n,
            stacked=True,
            color=colors[:4],
            label=keys,
      )
      plt.xlabel('Task')
      plt.ylabel('Score')
      plt.title('Tasks repartition among different values of top_k, top_p and temperature')
      plt.xticks(rotation=45, ha='right')
      plt.legend(title="Data Sources", fontsize='small', ncol=2)
      plt.tight_layout()
      plt.savefig(save_path, dpi=600)
      plt.show()
def plot_best_args(vizList:list[Visualization], save_path:str, n:int=5):
      """
      Plot the best arguments for a specific task across multiple visualizations.
      """
      data = get_best_tasks_args(vizList)[:n]
      keys = [get_name_from_path(viz.file_path) if get_name_from_path(viz.file_path) in list(map(lambda x: x[0], data)) else None for viz in vizList]

      final_data = []
      final_keys = []
      i = 0
      for j, v in enumerate(keys):
            if v is not None:
                  temp = vizList[j].df.select_dtypes(include='number').values
                  final_keys += [v] * len(temp)
                  final_data.append(*temp)
                  i+=1

      colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22'   # yellow-green
      ]

      plt.figure(figsize=(10, 6))
      plt.hist(
            [[i] for i in map(lambda x: x[1], data)],
            bins=5*n,
            stacked=True,
            color=colors[:4],
            label=list(map(lambda x: x[0], data)),
      )
      plt.xlabel('Task')
      plt.ylabel('Score')
      plt.title('Tasks repartition among different values of top_k, top_p and temperature')
      plt.xticks(rotation=45, ha='right')
      plt.legend(title="Data Sources", fontsize='small', ncol=2)
      plt.tight_layout()
      plt.savefig(save_path, dpi=600)
      plt.show()

def get_duplicates_similarity(vizList:list[Visualization]) -> list[tuple]:
      """
      Get the duplicates similarity across multiple visualizations.
      """
      duplicates = []
      for viz in vizList:
            viz.load_data()
            for i, row in viz.df.iterrows():
                  if row['text-classification'] > 0.9:  # Assuming a threshold of 0.9 for similarity
                        duplicates.append((i, row['text-classification']))
      return duplicates

vizualizer = [
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc0.csv', ascending=True),
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc1.csv', ascending=True),
      Visualization('data/sim_dataset-prompt/dataset0-similarityFunc2.csv', ascending=False),
      Visualization('data/sim_dataset-prompt/jaccard_without_embeddings.csv', ascending=False)
]
# for viz in vizualizer:
#       viz.load_data()
#       viz.top_values('text-classification')

# viz = Visualization('data/csv2/dataset0-similarityFunc0-prompt0.csv', ascending=True)
# viz.load_data()
# viz.hit_map()
# viz.bar_chart_threshold(6.5)

# print("ok")
#       viz.heat_map()


# for i, viz in enumerate(vizualizer):
#       viz.load_data()
      # viz.heat_map()
      # viz.get_repartition(simDistIndex=i)
      # viz.top_values('text-classification', n=5)

# vizualizer[0].load_data()
# data=" ".join(["text categorization", "document classification", "content labeling", "topic identification"]),
# vizualizer[0].zbar_chart_threshold(
#       data=data,
#       prompt4dataset=True
# )
# vizualizer[0].get_repartition(vizualizer[0].df.loc[data].to_dict(), nbins=10, simDistIndex=0, name="prompt4dataset")

def compute_args_rating():
      file_list = [
            "dataset1-top_k1-top_p0.5-temp0.5 (2).jsonl",
            "dataset1-top_k1-top_p0.5-temp0.5.jsonl",
            "dataset1-top_k2-top_p0.5-temp0.5 (2).jsonl",
            "dataset1-top_k2-top_p0.5-temp0.5.jsonl",
            "dataset1-top_k3-top_p0.5-temp0.5 (2).jsonl",
            "dataset1-top_k3-top_p0.5-temp0.5.jsonl",
      ]
      file_list = [
            "dataset2-top_k3-top_p0.2-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.2-temp0.5.jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5.jsonl",
            "dataset2-top_k3-top_p0.9-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.9-temp0.5.jsonl",
      ]

      file_list = [
            "dataset1-top_k2-top_p0.3-temp0.6.jsonl",
            "dataset1-top_k2-top_p0.4-temp0.4.jsonl",
            "dataset1-top_k2-top_p0.6-temp0.3.jsonl",
            "dataset1-top_k4-top_p0.3-temp0.6.jsonl",
            "dataset1-top_k4-top_p0.4-temp0.4.jsonl",
            "dataset1-top_k4-top_p0.6-temp0.3.jsonl",
            "dataset7-top_k2-top_p0.5-temp0.2.jsonl",
            "dataset7-top_k3-top_p0.2-temp0.5.jsonl",
            "dataset7-top_k3-top_p0.5-temp0.5.jsonl",
      ]
      file_list = [
            "dataset5-top_k2-top_p0.3-temp0.6.jsonl",
            "dataset5-top_k2-top_p0.4-temp0.4.jsonl",
            "dataset5-top_k2-top_p0.6-temp0.3.jsonl",
            "dataset5-top_k4-top_p0.3-temp0.6.jsonl",
            "dataset5-top_k4-top_p0.4-temp0.4.jsonl",
            "dataset5-top_k4-top_p0.6-temp0.3.jsonl",
      ]

      file_list = [
            "dataset1-seed200-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed218-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed242-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed254-top_k3-top_p0.6-temp0.3.jsonl",
      ]
      vizList = [Visualization(f'data/sim_tasks/diff_duplicates/dataset1/sim_over_tasks{file}.csv', ascending=True) for file in file_list]
      for viz in vizList : viz.load_data()

      best_res = get_best_tasks_args(vizList, average=True)
      best_res_std = get_best_tasks_args(vizList, average=False)
      highest_n = get_n_highest_task_values(vizList)

      print("\n------------------- SORTED MEANS -----------------\n")
      for v in best_res: print(v[0], v[1])
      print("\n------------------- SORTED STDS -----------------\n")
      for v in best_res_std: print(v[0], v[1])

      rankings = {v[0]: 0 for v in best_res}
      for i, v in enumerate(highest_n):
            if v[0][:56] in rankings.keys():
                  rankings[v[0][:56]] += i
            else:
                  rankings[v[0][:60]] = i
      print("\n------------------- RANKINGS -----------------\n")
      for i, v in rankings.items():
            print(i, v)

      print("\n------------------- SORTED VALUES -----------------\n")
      for v in highest_n:
            print(v[0], v[1])
      plot_best_values(vizList, "data/sim_tasks/diff_duplicates/dataset1/hist_scores_visual_no_duplicates.png", n=9)
      plot_best_values(vizList, "data/sim_tasks/diff_duplicates/dataset1/hq_hist_scores_visual_no_duplicates.png", n=35)
      plot_best_args(vizList, "data/sim_tasks/diff_duplicates/dataset1/hist_mean_scores_visual_no_duplicates.png", n=25)


      for i in best_res:
            print(i[0], i[1])
      

compute_args_rating()

# viz = Visualization('data/sim_tasks/task_selection/similarity_over_tasks.csv', ascending=True)
# viz.load_data()
# viz.heat_map()