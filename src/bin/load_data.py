import kagglehub
import os
import pandas as pd
import requests
import swifter
from util import mini

dirname = os.path.dirname(__file__)

NUM_IMAGES = 1000 if mini.is_mini() else 100_000

def main():
    data_path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

    recipes_path = os.path.join(data_path, 'recipes.parquet')
        
    df = pd.read_parquet(recipes_path).head(NUM_IMAGES)

    df['RecipeId'] = df['RecipeId'].astype(int)

    df.dropna(subset=['Images'], inplace=True)

    csv_out_path = os.path.join(dirname, '../../data/recipes.csv')
    img_out_dir = os.path.join(dirname, '../../data/dataset-b-images')

    if os.path.exists(csv_out_path):
        raise Exception(f"Path already exists {csv_out_path}")

    if os.path.exists(img_out_dir):
        raise Exception(f"Path already exists {img_out_dir}")

    def process_row(row):
        images = row['Images']
        recipe_id = row['RecipeId']

        if len(images) == 0:
            return False

        img_url = images[0]

        try:
            save_path = os.path.join(img_out_dir, f"{recipe_id}.jpg")
            # Send a GET request to the URL
            response = requests.get(img_url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes (4xx or 5xx)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Write the image to the specified location
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download the image for {recipe_id}: {e}")

            return False
        
        return True

    print('Starting up...')
    df = df[df.swifter.progress_bar(desc="Downloading images").apply(process_row, axis=1)]
    print('Done processing, writing final csv...')

    df[['RecipeId', 'Name']].to_csv(csv_out_path, index=False)
    print('Done!')
