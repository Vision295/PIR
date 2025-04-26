import pandas as pd
import matplotlib.pyplot as plt

class Visualization:
      def __init__(self, file_path):
            self.file_path = file_path
            self.df = None

      def load_data(self):
            self.df = pd.read_csv(self.file_path)
            #print(self.df.head())

      def plot_columns(self, x_column, y_column):
            if self.df is not None:
                  plt.plot(self.df[x_column], self.df[y_column])
                  plt.xlabel(x_column)
                  plt.ylabel(y_column)
                  plt.title(f'{y_column} vs {x_column}')
                  plt.show()
            else:
                  print("Data not loaded yet.")

      def top_values(self, column_name, n=5):
            if self.df is not None:
                  top = self.df[column_name].nlargest(n)
                  print(top)
            else:
                  print("Data not loaded yet.")

vizualizer = [
      Visualization('csv/dataset0-similarityFunc0.csv'),
      Visualization('csv/dataset0-similarityFunc1.csv'),
      Visualization('csv/dataset0-similarityFunc2.csv')
]
for viz in vizualizer:
      viz.load_data()
      viz.top_values('text-classification')