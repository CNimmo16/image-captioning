import torch
import math

class Attention(torch.nn.Module):
    def __init__(self, embed_dim: int, mask: bool = False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.mask = mask
        
        self.query_transform = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.key_transform = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.value_transform = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, queryable, keyable):
        Q = self.query_transform(queryable)
        K = self.key_transform(keyable)
        V = self.value_transform(keyable)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        
        if self.mask:
            # Create a causal mask (lower-triangular matrix).
            # Assumes that the query and key sequences are of equal length.
            seq_len = scores.size(-1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
            # Mask out future tokens by setting their scores to -infinity.
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        A = torch.nn.functional.softmax(scores, dim=-1)
        
        dE = torch.matmul(A, V)
        
        return dE
