import kagglehub
import os
import pandas as pd
import swifter
import aiohttp
import asyncio
import os

from util import mini

dirname = os.path.dirname(__file__)

NUM_IMAGES = 1000 if mini.is_mini() else 100_000

def main():
    data_path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

    recipes_path = os.path.join(data_path, 'recipes.parquet')
        
    df = pd.read_parquet(recipes_path).head(NUM_IMAGES)

    df['RecipeId'] = df['RecipeId'].astype(int)

    df.dropna(subset=['Images'], inplace=True)

    asyncio.run(download_images(df))

async def download_images(df):
    csv_out_path = os.path.join(dirname, '../../data/recipes.csv')
    img_out_dir = os.path.join(dirname, '../../data/dataset-b-images')

    if os.path.exists(csv_out_path):
        raise Exception(f"Path already exists {csv_out_path}")

    if os.path.exists(img_out_dir):
        raise Exception(f"Path already exists {img_out_dir}")

    print('Downloading images...')

    async def process_row(session, row, img_out_dir):
        images = row['Images']
        recipe_id = row['RecipeId']

        if len(images) == 0:
            df.drop(index=row.name, inplace=True)
            return

        img_url = images[0]

        try:
            save_path = os.path.join(img_out_dir, f"{recipe_id}.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            async with session.get(img_url) as response:
                with open(save_path, 'wb') as file:
                    file.write(await response.read())

        except Exception as e:
            print(f"Failed to download the image for {recipe_id}: {e}")
            df.drop(index=row.name, inplace=True)

    async with aiohttp.ClientSession() as session:
        tasks = [process_row(session, row, img_out_dir) for index, row in df.iterrows()]
        await asyncio.gather(*tasks)

    print('Done processing, writing final csv...')

    df[['RecipeId', 'Name']].to_csv(csv_out_path, index=False)
    print('Done!')
