import torch

from dataset import make_recipe_dataset
from bin.train import make_models, collate
from util import artifacts, devices, mini
from transformers import CLIPTokenizer
from matplotlib import pyplot as plt

def main():
    models = make_models()

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    device = devices.get_device()

    text_encoder = models['text_encoder']
    get_image_features = models['get_image_features']
    encoder_processor = models['encoder_processor']
    decoder = models['decoder']
    vocab_size = models['vocab_size']
    embed_dim = models['embed_dim']

    decoder_weights = artifacts.load_artifact('decoder-weights', 'model')
    decoder.load_state_dict(decoder_weights)

    decoder.eval()
    
    dataset = make_recipe_dataset(mini.is_mini())
        
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    max_length = tokenizer.model_max_length
    
    with torch.no_grad():
        for images, captions in eval_loader:    
            encoder_image_inputs = encoder_processor(images=images, return_tensors="pt").to(device)
    
            image_embeddings = get_image_features(**encoder_image_inputs).unsqueeze(1)
                
            token_ids = torch.tensor([tokenizer.bos_token_id]).to(device)

            for i in range(max_length):
                E = text_encoder(input_ids=token_ids).unsqueeze(0)
                E = torch.cat([image_embeddings, E], dim=1)
            
                logits = decoder(E)
                logits_text = logits[:, i + 1, :]
                logits_text = logits_text.reshape(-1, logits_text.shape[-1])
                
                # # Apply temperature scaling
                # logits = logits / temperature
                
                # # Top-k and top-p filtering
                # filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                                
                # # Sample next token
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
            print(desc)
            
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 15)  # Increased y-limit for more vertical space

            text = desc
            ax.text(5, 13, text, fontsize=12, ha="center", va="center", wrap=True,
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
            
            ax.imshow(images[0], extent=[1, 9, 2, 12], aspect='auto')

            ax.set_axis_off()
            plt.show()
