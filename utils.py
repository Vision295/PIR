
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

