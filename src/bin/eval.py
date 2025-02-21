import torch

from dataset import make_recipe_dataset
from bin.train import make_models, collate
from util import artifacts, devices, mini
from matplotlib import pyplot as plt
from inference import predict_dish_name
import numpy as np

NUM_IMAGES = 15

def main():
    models = make_models(eval=True)

    tokenizer = models['tokenizer']

    dataset = make_recipe_dataset(mini.is_mini())
        
    random_sampler = torch.utils.data.RandomSampler(dataset, num_samples=NUM_IMAGES)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=random_sampler, collate_fn=collate)
    
    max_length = tokenizer.model_max_length
    
    with torch.no_grad():
        def predict(images, captions):
            prediction = predict_dish_name(images[0], temperature=3, top_k=0)
            return images, captions, prediction

        predictions = [predict(images, captions) for images, captions in eval_loader]

        images = [images[0] for images, captions, prediction in predictions]
        captions = [prediction for images, captions, prediction in predictions]

        # Define grid size
        num_cols = 3
        num_rows = NUM_IMAGES // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

        # Flatten axes for easy iteration
        axes = axes.flatten()

        # Display images with captions
        for i, ax in enumerate(axes):
            ax.imshow(images[i], cmap='gray')  # Display image (change cmap if needed)
            ax.set_title(captions[i], fontsize=10)
            ax.axis('off')  # Hide axis

        # Adjust layout
        plt.tight_layout()
        plt.show()
