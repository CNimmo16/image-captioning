import torch
from models.decoder_layer import DecoderLayer

from util import devices
from util.typecheck import assert_shape

device = devices.get_device()

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        
    def forward(self, input_ids: torch.Tensor):
        E = self.embeddings(input_ids)

        return E
