import torch
from typing import Tuple

class Model(torch.nn.Module):

    def __init__(self, device: torch.device):
        super(Model, self).__init__()
        self.device = device
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')
        self.model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
        self.cls_head = torch.nn.Linear(self.model.pooler.dense.out_features, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, queries: Tuple[str], questions: Tuple[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: Query sentences (len=N)
            questions: Clarifying questions (len=N)

        Returns:
            seq_logits: Logits for (matching, not matching) (N, 2)
            seq_soft: Softmax probabilities for (matching, not matching) (N, 2)
        """
        # Tokenize every sample in the batch by concatenating the query and question
        encoded_dict = self.tokenizer(queries, questions, add_special_tokens=True, padding=True)

        # Convert to tensors
        indexed_tokens = torch.tensor(encoded_dict.data["input_ids"]).to(self.device)
        segments_ids = torch.tensor(encoded_dict.data["token_type_ids"]).to(self.device)
        attention_mask = torch.tensor(encoded_dict.data["attention_mask"]).float().to(self.device)

        # Forward pass (index = 0 not paraphrasing each other, index = 1 paraphrasing each other)
        out = self.model(indexed_tokens, token_type_ids=segments_ids, attention_mask=attention_mask)
        seq_logits = self.cls_head(out.pooler_output)

        # Compute softmax probabilties
        seq_soft = self.softmax(seq_logits)
        return seq_logits, seq_soft