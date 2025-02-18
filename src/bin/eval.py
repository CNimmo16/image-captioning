import torch

from dataset import Flickr30kDataset
from bin.train import make_models, collate
from util import artifacts, devices, mini

def main():
    models = make_models()
    
    device = devices.get_device()

    encoder = models['encoder']
    encoder_processor = models['encoder_processor']
    decoder = models['decoder']
    vocab_size = models['vocab_size']
    embed_dim = models['embed_dim']

    decoder_weights = artifacts.load_artifact('decoder-weights', 'model')
    decoder.load_state_dict(decoder_weights)

    decoder.eval()
    
    dataset = Flickr30kDataset()
    
    IMAGE_COUNT = 1

    if mini.is_mini():
        image = torch.tensor([999. for _ in range(embed_dim)]).to(device)
        caption = torch.tensor([0, 1, 2]).to(device)
        dataset = [(image, caption)]
    else:
        dataset = torch.utils.data.Subset(dataset, range(len(dataset) - IMAGE_COUNT, len(dataset)))
        
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)
    
    max_length = 5
    
    with torch.no_grad():
        for images, captions in eval_loader:
    
            if mini.is_mini():
                E = torch.stack(images).unsqueeze(1)
            else:
                encoder_image_inputs = encoder_processor(images=images, return_tensors="pt").to(device)
        
                image_embeddings = encoder.get_image_features(**encoder_image_inputs)
                E = image_embeddings.unsqueeze(1)
                
            start_token = torch.tensor([[0]]).to(device)
            start_token_embeddings = start_token.unsqueeze(2).repeat(1, 1, embed_dim)
            E = torch.cat([E, start_token_embeddings], dim=1)

            output_tokens = []

            for i in range(max_length):                                        
                logits = decoder(E)
                logits_text = logits[:, i + 1, :]
                logits_text = logits_text.reshape(-1, logits_text.shape[-1])
                
                # # Apply temperature scaling
                # logits = logits / temperature
                
                # # Top-k and top-p filtering
                # filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                                
                # # Sample next token
                probs = torch.nn.functional.softmax(logits_text, dim=-1).squeeze(1)
                
                # print(probs)
                if mini.is_mini():
                    next_token_idx = torch.argmax(probs, dim=-1)
                    
                    next_token = next_token_idx
                else:
                    next_token = torch.multinomial(probs, num_samples=1)
                                    
                next_token = torch.tensor([[next_token]]).to(device)

                if mini.is_mini():
                    next_token_embeddings = next_token.unsqueeze(2).repeat(1, 1, embed_dim)
                else:
                    next_token_embeddings = encoder.text_model(input_ids=next_token.unsqueeze(1)).last_hidden_state
                
                # Append token to input
                E = torch.cat([E, next_token_embeddings], dim=1)
                output_tokens.append(next_token)
                
                # # Stop if EOS token is generated
                # if next_token.item() == tokenizer.eos_token_id:
                #     break
            output_tokens = [int(token.item()) for token in output_tokens]
            
            # decoded = encoder_processor.batch_decode(output_tokens, skip_special_tokens=False)
            print(output_tokens)
