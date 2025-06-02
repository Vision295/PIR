import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


      def heat_map(self):
            if self.df is not None:
                  numeric_df = self.df.select_dtypes(include='number')

                  if numeric_df.empty:
                        print("Aucune colonne numérique à afficher.")
                        return

                  if self.file_path == 'data/data/sim_dataset-prompt/dataset0-similarityFunc0.csv':
                        numeric_df = numeric_df.round(0).astype(int)

                  # Nettoyer les descriptions des lignes (index)
                  def clean_description(desc, max_len=60):
                        desc = str(desc)
                        if desc.startswith("Task "):
                              desc = desc.split(". ", 1)[-1]  # Supprimer "Task X. "
                        return desc[:max_len] + "..." if len(desc) > max_len else desc

                  # Appliquer le nettoyage à l'index (lignes uniquement)
                  numeric_df.index = [clean_description(d) for d in numeric_df.index]


                  n_rows, n_cols = numeric_df.shape
                  cell_size_x = 1
                  cell_size_y = 0.3
                  figsize_x = max(10, n_cols * cell_size_x)
                  figsize_y = max(10, n_rows * cell_size_y)

                  plt.figure(figsize=(figsize_x, figsize_y))
                  if self.file_path == 'data/data/sim_dataset-prompt/dataset0-similarityFunc0.csv':
                        sns.heatmap(numeric_df, annot=True, cmap='viridis', fmt="d", annot_kws={"size": 8})
                  else:
                        sns.heatmap(numeric_df, annot=True, cmap='viridis', fmt=".2f", annot_kws={"size": 8})

                  plt.xticks([])

                  plt.title("Heatmap", fontsize=16)
                  plt.tight_layout()

                  save_path = self.file_path[:-3] + 'heatmap.png'
                  plt.savefig(save_path, dpi=300)
                  plt.close()
                  print(f"heat save to {save_path}")
            else:
                  print("Pas de donnees")

if __name__ == '__main__':

      data = [Visualization('data/data/sim_tasks/similarity_over_tasks/d1/datasets_and_tasks.csv', ascending=True),
              Visualization('data/data/sim_tasks/similarity_over_tasks/d5/datasets_and_tasks.csv', ascending=True)]
            # [Visualization('data/data/sim_tasks/similarity_over_tasks/d1/similarity_over_tasks.csv', ascending=True),
            # Visualization("data/data/sim_tasks/similarity_over_tasks/d1/clean_similarity_over_tasks.csv", ascending=True)]
            # [Visualization('data/data/sim_tasks/similarity_over_tasks/d5/similarity_over_tasks.csv', ascending=True),
            # Visualization("data/data/sim_tasks/similarity_over_tasks/d5/clean_similarity_over_tasks.csv", ascending=True)]

      for file in data:
            file.load_data()
            file.heat_map()


