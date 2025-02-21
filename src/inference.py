import torch

from dataset import make_recipe_dataset
from bin.train import make_models, collate
from util import artifacts, devices, mini
from matplotlib import pyplot as plt

_artifacts = None

def preload_artifacts():
    global _artifacts

    if _artifacts is not None:
        return _artifacts

    _artifacts = {
        'decoder_weights': artifacts.load_artifact('decoder-weights', 'model')
    }

    return _artifacts

def top_k_top_p_filtering(logits, top_k, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Remove tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    if top_k > 0:
        sorted_indices_to_remove[:, top_k:] = True
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float("Inf")
    return logits

def predict_dish_name(image, temperature=1.0, top_k=100, top_p=1.0):
    models = make_models(eval=True)
    
    device = devices.get_device()

    tokenizer = models['tokenizer']
    encoder = models['encoder']
    encoder_processor = models['encoder_processor']
    decoder = models['decoder']
    vocab_size = models['vocab_size']
    embed_dim = models['embed_dim']

    decoder_weights = preload_artifacts()['decoder_weights']
    decoder.load_state_dict(decoder_weights)

    decoder.eval()
    
    max_length = tokenizer.model_max_length
    
    with torch.no_grad():    
        encoder_image_inputs = encoder_processor(images=[image], return_tensors="pt").to(device)

        image_embeddings = encoder.get_image_features(**encoder_image_inputs).unsqueeze(1)
            
        token_ids = torch.tensor([tokenizer.bos_token_id]).to(device)

        for i in range(max_length):
            E = encoder.text_model(input_ids=token_ids).last_hidden_state
            E = torch.cat([image_embeddings, E], dim=1)             
        
            logits = decoder(E)

            logits_text = logits[:, i + 1, :]
            logits_text = logits_text.reshape(-1, logits_text.shape[-1])
            
            # Apply temperature scaling
            logits_text = logits_text / temperature
            
            # Apply top-k and top-p filtering
            logits_text = top_k_top_p_filtering(logits_text, top_k=top_k, top_p=top_p)
            
            # Sample next token
            probs = torch.nn.functional.softmax(logits_text, dim=-1).squeeze(1)
            next_token = torch.multinomial(probs, num_samples=1)
                                
            next_token = torch.tensor([next_token]).to(device)
            
            # Append token to input
            token_ids = torch.cat([token_ids, next_token], dim=0)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

        token_ids = [int(token.item()) for token in token_ids]
        
        desc = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        return desc
