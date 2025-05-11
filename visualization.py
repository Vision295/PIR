import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
      def __init__(self, file_path, ascending=True):
            self.file_path = file_path
            self.df = None
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

      def hit_map(self):
            if self.df is not None:
                  numeric_df = self.df.select_dtypes(include='number')
                  plt.figure(figsize=(20, 20))
                  sns.heatmap(numeric_df, annot=True, cmap='viridis')
                  plt.title("Heatmap")
                  plt.tight_layout()
                  plt.savefig('heatmap.png')
                  print("Heatmap saved to heatmap.png")
            else:
                  print("Data not loaded yet.")
      
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
                  print("Heatmap saved to barchart.png")
            else:
                  print("Données non chargées.")



vizualizer = [
      Visualization('csv/dataset0-similarityFunc0.csv', ascending=True),
      Visualization('csv/dataset0-similarityFunc1.csv', ascending=True),
      Visualization('csv/dataset0-similarityFunc2.csv', ascending=False)
]
# for viz in vizualizer:
#       viz.load_data()
#       viz.top_values('text-classification')

viz = Visualization('csv/dataset0-similarityFunc1.csv', ascending=True)
viz.load_data()
viz.hit_map()
viz.bar_chart_threshold(6.5)

print("ok")


