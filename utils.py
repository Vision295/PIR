from torch.nn.functional import cosine_similarity, pairwise_distance
from torch import Tensor
import torch

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



file_names = {

      1:[ 
            "dataset2-top_k1-top_p0.5-temp0.5 (2).jsonl",
            "dataset2-top_k1-top_p0.5-temp0.5.jsonl",
            "dataset2-top_k2-top_p0.5-temp0.5 (2).jsonl",
            "dataset2-top_k2-top_p0.5-temp0.5.jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5.jsonl",
      ],

      2:[
            "dataset2-top_k3-top_p0.2-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.2-temp0.5.jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.5-temp0.5.jsonl",
            "dataset2-top_k3-top_p0.9-temp0.5 (2).jsonl",
            "dataset2-top_k3-top_p0.9-temp0.5.jsonl",
      ],

      3:[
            "dataset5-top_k2-top_p0.5-temp0.2 (2).jsonl",
            "dataset5-top_k2-top_p0.5-temp0.2.jsonl",
            "dataset5-top_k2-top_p0.5-temp0.5 (2).jsonl",
            "dataset5-top_k2-top_p0.5-temp0.5.jsonl",
            "dataset5-top_k2-top_p0.5-temp0.9 (2).jsonl",
            "dataset5-top_k2-top_p0.5-temp0.9.jsonl",
      ],

      4:[
            "dataset1-top_k2-top_p0.3-temp0.6.jsonl",
            "dataset1-top_k2-top_p0.4-temp0.4.jsonl",
            "dataset1-top_k2-top_p0.6-temp0.3.jsonl",
            "dataset1-top_k4-top_p0.3-temp0.6.jsonl",
            "dataset1-top_k4-top_p0.4-temp0.4.jsonl",
            "dataset1-top_k4-top_p0.6-temp0.3.jsonl",
            "dataset7-top_k2-top_p0.5-temp0.2.jsonl",
            "dataset7-top_k3-top_p0.2-temp0.5.jsonl",
            "dataset7-top_k3-top_p0.5-temp0.5.jsonl",
      ],

      5:[
            "dataset5-seed206-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset5-seed218-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset5-seed242-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset5-seed254-top_k3-top_p0.6-temp0.3.jsonl",
      ],

      6:[
            "dataset1-seed200-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed218-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed242-top_k3-top_p0.6-temp0.3.jsonl",
            "dataset1-seed254-top_k3-top_p0.6-temp0.3.jsonl",
      ],

      7:[
            "dataset5-top_k2-top_p0.3-temp0.6.jsonl",
            "dataset5-top_k2-top_p0.4-temp0.4.jsonl",
            "dataset5-top_k2-top_p0.6-temp0.3.jsonl",
            "dataset5-top_k4-top_p0.3-temp0.6.jsonl",
            "dataset5-top_k4-top_p0.6-temp0.3.jsonl",
      ],
}


def get_name_from_path(path:str) -> str:
      """
      Extracts the name of the file from a given path.
      """
      return path.rsplit('/', 1)[-1]


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