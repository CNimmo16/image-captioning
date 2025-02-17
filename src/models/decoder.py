import torch
from models.decoder_layer import DecoderLayer

from util import devices
from util.typecheck import assert_shape

device = devices.get_device()

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size: int, decoder_layers: int, embed_dim: int, mlp_hidden_dim: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        self.embed_dim = embed_dim
        
        self.text_projection = torch.nn.Linear(embed_dim, embed_dim)
        
        self.layers = torch.nn.ModuleList([DecoderLayer(embed_dim, mlp_hidden_dim) for _ in range(decoder_layers)])
        
        self.final_projection = torch.nn.Linear(embed_dim, vocab_size)
        
    def forward(self, E: torch.Tensor):
        batch_size = E.shape[0]
                
        sequence_length = E.shape[1]
        
        for layer in self.layers:
            residual = E
            O = layer(E)
            E = residual + O

        logits = self.final_projection(E)
        assert_shape(logits, (batch_size, sequence_length, self.vocab_size))
        
        return logits
