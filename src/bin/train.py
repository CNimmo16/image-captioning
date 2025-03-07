import torch
import tqdm
import random
import os
from pathlib import Path
import wandb
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import gc
import time

from util.typecheck import assert_shape
from util import devices, mini, debug, constants, artifacts
from models.decoder import Decoder
from dataset import make_flickr_dataset, make_recipe_dataset, validate_data
import evaluate

torch.manual_seed(16)
random.seed(16)

is_mini = mini.is_mini()

if is_mini:
    print(f"INFO: Using mini mode")

device = devices.get_device()

print(f"INFO: Using device {device}")

DECODER_LAYERS = 10
MLP_HIDDEN_DIM = 128

BATCH_SIZE = 128
LEARNING_RATE = 0.0003
DROPOUT = 0.15
EPOCHS = 20
EARLY_STOP_AFTER_EPOCHS = float('inf')
    
hyperparams = {
    "is_mini": is_mini,
    "decoder_layers": DECODER_LAYERS,
    "mlp_hidden_dim": MLP_HIDDEN_DIM,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "architecture_version": 4 # change this manually when the model's architecture changes
}

models = None

def make_models(eval: bool):
    global models
    if models:
        return models

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    if eval:
        for param in encoder.parameters():
            # freeze whole model for evaluation
            param.requires_grad = False
    else:
        for param in encoder.vision_model.parameters():
            # freeze vision model while finetuning text
            param.requires_grad = False

    vocab_size = encoder.text_model.config.vocab_size
    
    embed_dim = encoder.text_model.config.hidden_size

    models = {
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'encoder': encoder,
        'tokenizer': tokenizer,
        'encoder_processor': CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
        'decoder': Decoder(vocab_size, DECODER_LAYERS, embed_dim, MLP_HIDDEN_DIM, DROPOUT).to(device)
    }

    return models
    
def main():
    validate_data()

    print('Loading models...')

    models = make_models(eval=False)
    
    tokenizer = models['tokenizer']
    encoder = models['encoder']
    encoder_processor = models['encoder_processor']
    decoder = models['decoder']
    vocab_size = models['vocab_size']
    embed_dim = models['embed_dim']

    print('Initialising dataset...')

    dataset = make_recipe_dataset(mini.is_mini())

    print('Loading BLEU for evaluation...')

    bleu = evaluate.load("bleu")

    print('Initialising training...')
    
    if mini.is_mini():
        train = dataset
        val = dataset
    else:
        val_split = 0.1
        max_val_count = 5000
        data_count = len(dataset)
        val_count = round(min(data_count * val_split, max_val_count))
        train = torch.utils.data.Subset(dataset, range(data_count - val_count))
        val = torch.utils.data.Subset(dataset, range(data_count - val_count, data_count))

    indices = list(range(len(train)))

    # Shuffle the indices
    random.shuffle(indices)

    # Create DataLoaders for training and validation
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(indices), # perform shuffle once at start to allow caching batches across epochs
        collate_fn=collate,
        shuffle=False
    )

    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=True)

    wandb.init(project='image-captioning', name='image-decoder', config={
        "training_data_size": len(train_loader),
    } | hyperparams)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(list(decoder.parameters()) + list(encoder.parameters()), lr=LEARNING_RATE, betas=(0.9, 0.98))
    
    for layer in decoder.layers:
        for param in layer.attention.parameters():
            if len(param.shape) > 1:  # Only apply to weight matrices, not biases
                torch.nn.init.xavier_uniform_(param)
    
    val_loss_failed_to_improve_for_epochs = 0
    best_val_loss = float('inf')
    best_state_dict = None
    
    train_cache = {}
    val_cache = {}
    
    for epoch in range(EPOCHS):
        suffix = " (initial validation)" if epoch == 0 else ""
        print(f"Epoch {epoch} {suffix}")

        if epoch > 0:
            decoder.train()
            epoch_train_loss = 0

            for batch_idx, (images, captions) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f"> training"):
                batch_size = len(images)
                
                optimizer.zero_grad()
                
                if batch_idx in train_cache:
                    cached = train_cache[batch_idx]
                    image_embeddings = cached['image_embeddings']
                    encoder_text_inputs = cached['text_inputs']
                    text_embeddings = cached['text_embeddings']
                else:
                    encoder_image_inputs = encoder_processor(images=images, return_tensors="pt").to(device)
                    with torch.no_grad():
                        image_embeddings = encoder.get_image_features(**encoder_image_inputs)
                    image_embeddings = image_embeddings.unsqueeze(1)

                    try:
                        encoder_text_inputs = encoder_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
                    except:
                        print('ERROR cant encode, received:', captions)
                        raise Exception()
                    with torch.no_grad():
                        text_embeddings = encoder.text_model(**encoder_text_inputs).last_hidden_state

                    train_cache[batch_idx] = {
                        'image_embeddings': image_embeddings,
                        'text_inputs': encoder_text_inputs,
                        'text_embeddings': text_embeddings
                    }

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
        else:
            epoch_train_loss = 0

        decoder.eval()
        epoch_val_loss = 0
        epoch_bleu_score = 0
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
                
                predicted_token_ids = torch.argmax(logits_text, dim=-1)

                predictions = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

                references = [c.lower() for c in captions]
                predictions = [';;;;;;;' if x == '' else x for x in predictions] # avoid empty strings which bleu cant deal with

                results = bleu.compute(predictions=predictions, references=references)
                
                epoch_bleu_score += results['bleu']

                epoch_val_loss += loss.item()

        epoch_grad_norms = debug.compute_grad_norms(decoder)
        
        vanishing, exploding = debug.detect_gradient_issues(epoch_grad_norms, vanish_thresh=1e-6, explode_thresh=10.0)
        
        vanishing_gradients = len(vanishing)
        exploding_gradients = len(exploding)
        
        epoch_train_loss = epoch_train_loss / len(train_loader)
        epoch_val_loss = epoch_val_loss / len(val_loader)
        epoch_bleu_score = epoch_bleu_score / len(val_loader)
        
        if exploding_gradients > 0 or vanishing_gradients > 0:
            print('Clipping gradients...')
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                            
        print(f"Train loss: {epoch_train_loss}, val loss: {epoch_val_loss}, bleu score: {epoch_bleu_score}\n")

        wandb.log({
            'epoch': epoch,
            'train-loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'bleu_score': epoch_bleu_score,
            'vanishing_gradients': vanishing_gradients,
            'exploding_gradients': exploding_gradients
        })
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            val_loss_failed_to_improve_for_epochs = 0
            best_state_dict = decoder.state_dict()
        else:
            val_loss_failed_to_improve_for_epochs += 1

        if not mini.is_mini():
            epoch_save_path = os.path.join(constants.DATA_PATH, f"epoch-weights/decoder-weights_epoch-{epoch}.generated.pt")
            torch.save(decoder.state_dict(), epoch_save_path)

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
