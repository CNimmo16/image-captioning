import kagglehub
import os
import pandas as pd
import aiohttp
import asyncio
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO

from util import mini

dirname = os.path.dirname(__file__)

NUM_IMAGES = 1000 if mini.is_mini() else 300_000

CHUNK_SIZE = 10000

def main():
    csv_out_path = os.path.join(dirname, '../../data/recipes.csv')
    img_out_dir = os.path.join(dirname, '../../data/dataset-b-images')

    if os.path.exists(csv_out_path):
        raise Exception(f"Path already exists {csv_out_path}")

    if os.path.exists(img_out_dir):
        raise Exception(f"Path already exists {img_out_dir}")
    
    async def download_images(chunk):

        async def process_row(session, row, img_out_dir, pbar):
            images = row['Images']
            recipe_id = row['RecipeId']

            img_url = images[0]

            try:
                save_path = os.path.join(img_out_dir, f"{recipe_id}.jpg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                async with session.get(img_url) as response:
                    with open(save_path, 'wb') as file:
                        img_data = await response.read()
                        try:
                            img = Image.open(BytesIO(img_data))
                            img.verify()
                            file.write(img_data)
                        except Exception as e:
                            print(f"Image for {recipe_id} is invalid ({e}), removing from dataframe...")
                            df.drop(index=row.name, inplace=True)

            except Exception as e:
                print(f"Failed to download the image for {recipe_id}: {e}")
                df.drop(index=row.name, inplace=True)
            finally:
                pbar.update(1)

        async with aiohttp.ClientSession() as session:
            with tqdm(total=len(chunk), desc="Downloading images", unit="file") as pbar:
                tasks = [process_row(session, row, img_out_dir, pbar) for index, row in chunk.iterrows()]
                await asyncio.gather(*tasks)

    data_path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

    recipes_path = os.path.join(data_path, 'recipes.parquet')
        
    df = pd.read_parquet(recipes_path).head(NUM_IMAGES)

    print(f"Loaded {len(df)} recipes")

    df['RecipeId'] = df['RecipeId'].astype(int)

    df.dropna(subset=['Images'], inplace=True)

    df = df[df['Images'].apply(lambda x: len(x) > 0)]

    print(f"Removed recipes with no images: {len(df)} remaining")

    asyncio.run(download_images(df))

    print('Done processing, writing final csv...')

    print('Done!')
