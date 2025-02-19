import torch
import pandas as pd
import PIL
import os
import numpy as np
import kagglehub

dirname = os.path.dirname(__file__)

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        data_path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
        self.image_dir = os.path.join(data_path, 'flickr30k_images/flickr30k_images')

        captions_file = os.path.join(data_path, 'flickr30k_images/results.csv')
        
        self.transform = transform
        
        self.captions_df = pd.read_csv(captions_file, sep='|', names=['image_name', 'caption_idx', 'caption_text'], skiprows=1)
        
        broken_idx = 19999 # randomnly has no separator in train set

        self.captions_df.drop(index=broken_idx, inplace=True)
        self.captions_df['caption_idx'] = self.captions_df['caption_idx'].astype(np.int32)
        
        self.captions_df['image_name'] = self.captions_df['image_name'].str.strip()
        self.captions_df['caption_text'] = self.captions_df['caption_text'].str.strip()

    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        
        image_name = row['image_name']
        
        image_path = os.path.join(self.image_dir, image_name)
        
        image = PIL.Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption_text = row['caption_text']
            
        return image, caption_text

def make_flickr_dataset(mini: bool):
    dataset = Flickr30kDataset()

    if mini:
        dataset = torch.utils.data.Subset(dataset, range(0, 40, 5))

    return dataset
