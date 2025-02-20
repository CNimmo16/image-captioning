from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from typing import Annotated
import logging
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

import inference

import os
dirname = os.path.dirname(__file__)

inference.preload_artifacts()

app = FastAPI()

def base64_to_pil(image_base64: str) -> Image.Image:
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    return image

class Body(BaseModel):
    image: str

@app.post("/predict")
async def root(request: Request, body: Body):
    image = base64_to_pil(body.image)

    prediction = inference.predict_dish_name(image)
    return {
        'dish_name': prediction
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
