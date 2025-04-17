from transformers import AutoTokenizer, AutoModel
import torch

class PromptConverter:

      def __init__(self, *args):
            self.inputs = [*args]

            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

      def tokenize(self):
            self.tokenizedInputs = list(map(
                  lambda x: self.tokenizer(x, return_tensors="pt", truncation=False),
                  self.inputs
            ))
      
      def generate_embeddings(self) -> str:
            self.tokenize()
            # Generate embeddings
            with torch.no_grad():  # No gradient calculation needed for inference
                  self.outputs = list(map(
                        lambda x: self.model(**x),
                        self.tokenizedInputs
                  ))

            self.embeddings = list(map(
                  lambda x: x.last_hidden_state.mean(dim=1),  # Mean pooling
                  self.outputs
            ))

            return f"Embeddings shapes: {[i.shape for i in self.embeddings][:2]} Sample embedding: {[i[0][:5] for i in self.embeddings][:2]}"

      def compute_similarity(self, similarityFunction) -> str:

            self.similarities = list(map(
                  similarityFunction,
                  zip(self.embeddings, self.embeddings[1:])
            ))

            return f"Similarities shapes: {[i.shape for i in self.similarities][:2]} Sample similarity: {[i[0] for i in self.similarities][:2]}"