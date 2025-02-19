import torch
import tqdm
import random
import os
from pathlib import Path
import wandb
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import gc

import func
import func.preprocessing
from util.typecheck import assert_shape
from util import devices, mini, debug, constants, artifacts
from models.decoder import Decoder
from dataset import make_flickr_dataset, make_recipe_dataset

torch.manual_seed(16)
random.seed(16)

is_mini = mini.is_mini()

if is_mini:
    print(f"INFO: Using mini mode")

device = devices.get_device()

DECODER_LAYERS = 10
MLP_HIDDEN_DIM = 128

BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.15
EPOCHS = 100
EARLY_STOP_AFTER_EPOCHS = 5
    
hyperparams = {
    "is_mini": is_mini,
    "decoder_layers": DECODER_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "architecture_version": 1 # change this manually when the model's architecture changes
}

def make_models():
    encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    encoder.requires_grad_(False)

    vocab_size = encoder.text_model.config.vocab_size
    
    embed_dim = encoder.text_model.config.hidden_size

    return {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'encoder': encoder,
        'encoder_processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'decoder': Decoder(vocab_size, DECODER_LAYERS, embed_dim, MLP_HIDDEN_DIM, DROPOUT).to(device)
    }
    
def main():
    models = make_models()
    
    encoder = models['encoder']
    encoder_processor = models['encoder_processor']
    decoder = models['decoder']
    vocab_size = models['vocab_size']
    embed_dim = models['embed_dim']

    dataset = make_recipe_dataset(mini.is_mini())
    
    if mini.is_mini():
        train = dataset
        val = dataset
    else:
        val_split = 0.2
        data_count = len(dataset)
        train = torch.utils.data.Subset(dataset, range(int(data_count * (1 - val_split))))
        val = torch.utils.data.Subset(dataset, range(int(data_count * (1 - val_split)), data_count))

    # Create DataLoaders for training and validation
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    wandb.init(project='image-captioning', name='image-decoder', config={
        "training_data_size": len(train_loader),
    } | hyperparams)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98))
    
    for layer in decoder.layers:
        for param in layer.attention.parameters():
            if len(param.shape) > 1:  # Only apply to weight matrices, not biases
                torch.nn.init.xavier_uniform_(param)
    
    val_loss_failed_to_improve_for_epochs = 0
    best_val_loss = float('inf')
    best_state_dict = None
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        decoder.train()
        epoch_train_loss = 0
        for images, captions in tqdm.tqdm(train_loader, desc=f"> training"):
            batch_size = len(images)
            
            optimizer.zero_grad()
            
            encoder_image_inputs = encoder_processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                image_embeddings = encoder.get_image_features(**encoder_image_inputs)
            image_embeddings = image_embeddings.unsqueeze(1)

            encoder_text_inputs = encoder_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                text_embeddings = encoder.text_model(**encoder_text_inputs).last_hidden_state

            E = torch.cat([image_embeddings, text_embeddings], dim=1)
                            
            logits = decoder(E)
            
            assert_shape(logits, (batch_size, '*', vocab_size))
            
            # remove image embedding
            logits_text = logits[:, :-1, :]
            
            loss = criterion(
                logits_text.reshape(-1, logits_text.shape[-1]),
                encoder_text_inputs['input_ids'].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            gc.collect()
            torch.cuda.empty_cache()

        decoder.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, captions in tqdm.tqdm(val_loader, desc=f"> validating"):
                batch_size = len(images)
                
                encoder_image_inputs = encoder_processor(images=images, return_tensors="pt").to(device)
                image_embeddings = encoder.get_image_features(**encoder_image_inputs)
                image_embeddings = image_embeddings.unsqueeze(1)
                
                encoder_text_inputs = encoder_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
                text_embeddings = encoder.text_model(**encoder_text_inputs).last_hidden_state
                            
                E = torch.cat([image_embeddings, text_embeddings], dim=1)
                                
                logits = decoder(E)
                
                assert_shape(logits, (batch_size, '*', vocab_size))
                
                # remove image embedding
                logits_text = logits[:, :-1, :]
                
                loss = criterion(
                    logits_text.reshape(-1, logits_text.shape[-1]),
                    encoder_text_inputs['input_ids'].reshape(-1)
                )

                epoch_val_loss += loss.item()

        epoch_grad_norms = debug.compute_grad_norms(decoder)
        
        vanishing, exploding = debug.detect_gradient_issues(epoch_grad_norms, vanish_thresh=1e-6, explode_thresh=10.0)
        
        vanishing_gradients = len(vanishing)
        exploding_gradients = len(exploding)
        
        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_val_loss = epoch_val_loss / len(val_loader)
        
        if exploding_gradients > 0 or vanishing_gradients > 0:
            print('Clipping gradients...')
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                            
        print(f"Train loss: {epoch_train_loss}, val loss: {epoch_val_loss}\n")

        wandb.log({
            'epoch': epoch + 1,
            'train-loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'vanishing_gradients': vanishing_gradients,
            'exploding_gradients': exploding_gradients
        })
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            val_loss_failed_to_improve_for_epochs = 0
            best_state_dict = decoder.state_dict()

            if not mini.is_mini():
                epoch_save_path = os.path.join(constants.DATA_PATH, f"epoch-weights/decoder-weights_epoch-{epoch}.generated.pt")
                torch.save(best_state_dict, epoch_save_path)
        else:
            val_loss_failed_to_improve_for_epochs += 1

        if val_loss_failed_to_improve_for_epochs == EARLY_STOP_AFTER_EPOCHS:
            print(f"Validation loss failed to improve for {EARLY_STOP_AFTER_EPOCHS} epochs. Early stopping now.")
            break
    
    model_save_path = os.path.join(constants.DATA_PATH, 'decoder-weights.generated.pt')
    torch.save(best_state_dict, model_save_path)
    artifacts.store_artifact('decoder-weights', 'model', model_save_path, hyperparams)

    wandb.finish()


def collate(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)
