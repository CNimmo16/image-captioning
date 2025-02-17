import torch
from models.decoder import Decoder
from util import devices

device = devices.get_device()

class ImageCaptioner(torch.nn.Module):
    
    def __init__(self, decoder_layers: int, mlp_hidden_dim: int):
        super().__init__()

        self.encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

        self.encoder_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.decoder = Decoder(self.encoder, self.encoder_processor, decoder_layers, self.encoder.vision_embed_dim, mlp_hidden_dim)
        
    def forward(self, images: list, texts: list):        
        encoder_inputs = self.encoder_processor(images=images, return_tensors="pt").to(device)
        
        vision_outputs = self.encoder.vision_model(**encoder_inputs)
        
        cls_token = vision_outputs.last_hidden_state[:, 0, :]
        
        decoder_outputs = self.decoder(texts, cls_token)
        return decoder_outputs
