import torch
from models.attention import Attention

class DecoderLayer(torch.nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int):
        super().__init__()
        
        self.norm_1 = torch.nn.LayerNorm(embed_dim)
        
        self.attention = Attention(embed_dim, mask=True)
        
        self.norm_2 = torch.nn.LayerNorm(embed_dim)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, embed_dim)
        )
        
    def forward(self, E: torch.Tensor):

        E = self.norm_1(E)
        
        dE = self.attention(E, E)
        
        E = E + dE
        
        E = E + self.mlp(E)
        
        E = self.norm_2(E)
        
        return E
