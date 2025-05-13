from torch.nn.functional import cosine_similarity, pairwise_distance
from torch import Tensor
import torch
from scipy.stats import wasserstein_distance

from scipy.stats import wasserstein_distance


"""
      csv_writer : 
      [
            {"descriptions": list of all the elements of datasetDescriptions},
            {datasetDescriptions[0]: [similarity between datasetDescriptions[0] and datasetDescriptions[0], ... up to datasetDescriptions[size]]},
            {datasetDescriptions[1]: [similarity between datasetDescriptions[1] and datasetDescriptions[0], ... up to datasetDescriptions[size]]},
            ...
            {datasetDescriptions[size]: [similarity between datasetDescriptions[n] and datasetDescriptions[0], ... up to datasetDescriptions[size]]}
      ]

"""
csv_writer = list[dict[str, list[str]] | dict[str, list[float]]]

similarityFunctions = [
      lambda x: pairwise_distance(x[0], x[1], p=0.5),
      lambda x: pairwise_distance(x[0], x[1], p=2),
      lambda x: cosine_similarity(x[0], x[1], dim=0),
      lambda x: jaccard_index_for_embeddings(x[0], x[1]),
      lambda x: wasserstein_distance(x[0], x[1]),
]

repartitionThresholds = [
      [30000, 32500, 35000, 37500],
]

def jaccard_index_for_embeddings(embedding1:torch.Tensor, embedding2:torch.Tensor, threshold:float=0.5) -> torch.Tensor:
      """
      On binarise les embeddings et on applique l'index de jaccard sur les embeddings binaires
      """
      binary1 = (embedding1 > threshold).float()
      binary2 = (embedding2 > threshold).float()

      intersection = torch.sum(binary1 * binary2)
      union = torch.sum((binary1 + binary2) > 0)

      return intersection / union if union > 0 else torch.tensor(0.0)

def remove_duplicates(inputFile:str = "datasetdetails.jsonl", outputFile:str= "datasetdetails_cleaned.jsonl"):

      # Use a set to track unique lines
      unique_lines = set()

      # Open the input file and process each line
      with open(inputFile, "r", encoding="utf-8") as infile, open(outputFile, "w", encoding="utf-8") as outfile:
            for line in infile:
                  # Strip whitespace and check if the line is unique
                  line = line.strip()
                  if line not in unique_lines:
                        unique_lines.add(line)
                        outfile.write(line + "\n")


#remove_duplicates("datasetdetails.jsonl")


print(
      cosine_similarity(
            Tensor([1.0, 2.0]),
            Tensor([2.0, 3.0]),
            dim=0
      )
)

print(
      pairwise_distance(
            Tensor([1.0, 2.0]),
            Tensor([2.0, 3.0]),
            p=2
      )     
)