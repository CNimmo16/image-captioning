import torch
import pandas as pd
import PIL
import os
import numpy as np
import shutil
import requests
import random
import kagglehub
import sys

dirname = os.path.dirname(__file__)

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        data_path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
        self.image_dir = os.path.join(data_path, 'flickr30k_images/flickr30k_images')

        captions_file = os.path.join(data_path, 'flickr30k_images/results.csv')
        
        self.transform = transform
        
        self.df = pd.read_csv(captions_file, sep='|', names=['image_name', 'caption_idx', 'caption_text'], skiprows=1)
        
        broken_idx = 19999 # randomnly has no separator in train set

        self.df.drop(index=broken_idx, inplace=True)
        self.df['caption_idx'] = self.df['caption_idx'].astype(np.int32)
        
        self.df['image_name'] = self.df['image_name'].str.strip()
        self.df['caption_text'] = self.df['caption_text'].str.strip()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
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

class RecipeDatasetA(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        data_path = kagglehub.dataset_download("pes12017000148/food-ingredients-and-recipe-dataset-with-images")

        self.image_dir = os.path.join(data_path, 'Food Images/Food Images')

        captions_file = os.path.join(data_path, 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
        
        self.transform = transform
        
        self.df = pd.read_csv(captions_file, sep=',')
        
        self.df.rename({
            'Image_Name': 'image_name',
            'Title': 'caption_text'
        }, axis=1, inplace=True)
        
        self.df = self.df[['image_name', 'caption_text']]

        self.df = self.df[self.df['image_name'] != '#NAME?']
        self.df = self.df[self.df['caption_text'] != '']
        self.df.dropna(inplace=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image_name = row['image_name']
        
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        
        image = PIL.Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption_text = row['caption_text']
            
        return image, caption_text

class RecipeDatasetB(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.recipes_path = os.path.join(dirname, '../data/recipes.csv')
        
        self.transform = transform
        
        self.df = pd.read_csv(self.recipes_path)

        self.image_dir = os.path.join(dirname, '../data/dataset-b-images')

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = f"{row['RecipeId']}.jpg"
        
        image_path = os.path.join(self.image_dir, img_name)
        
        image = PIL.Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        caption_text = row['Name']
            
        return image, caption_text

def make_recipe_dataset(mini: bool):
    datasetA = RecipeDatasetA()
    datasetB = RecipeDatasetB()
    dataset = torch.utils.data.ConcatDataset([datasetA, datasetB])

    if mini:
        dataset = torch.utils.data.Subset(dataset, range(5))

    return dataset

def validate_data():
    print('Verifying all data is present and valid...')
    datasetA = RecipeDatasetA()
    for index, row in datasetA.df.iterrows():
        img_name = f"{row['image_name']}.jpg"
        image_path = os.path.join(datasetA.image_dir, img_name)
        
        validate_image('dataset A', row['caption_text'], image_path)
    print('> All data present for dataset A ✅')
    
    datasetB = RecipeDatasetB()
    for index, row in datasetB.df.iterrows():
        img_name = f"{row['RecipeId']}.jpg"
        image_path = os.path.join(datasetB.image_dir, img_name)

        validate_image('dataset B', row['RecipeId'], image_path)
    print('> All data present for dataset B ✅')

def validate_image(dataset_alias, recipe_name, image_path):
    if not os.path.exists(image_path):
        print(f"❗ Image missing from dataset A for recipe with name {row['caption_text']} - {image_path} does not exist")
        sys.exit(1)
    try:
        img = PIL.Image.open(image_path)
        img.verify()
    except Exception as e:
        print(f"❗ Image missing from dataset A for recipe with name {row['caption_text']} - {image_path} is invalid")
        sys.exit(1)
