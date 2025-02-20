import torch

from dataset import make_recipe_dataset
from bin.train import make_models, collate
from util import artifacts, devices, mini
from matplotlib import pyplot as plt
from inference import predict_dish_name

def main():
    models = make_models(eval=True)

    tokenizer = models['tokenizer']

    dataset = make_recipe_dataset(mini.is_mini())
        
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    max_length = tokenizer.model_max_length
    
    with torch.no_grad():
        for images, captions in eval_loader:
            
            prediction = predict_dish_name(images[0])
            
            fig, ax = plt.subplots(figsize=(6, 8))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 15)  # Increased y-limit for more vertical space

            ax.text(5, 13, prediction, fontsize=12, ha="center", va="center", wrap=True,
                bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.5'))
            
            ax.imshow(images[0], extent=[1, 9, 2, 12], aspect='auto')

            ax.set_axis_off()
            plt.show()
