from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse
from typing import Annotated
import logging
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image

import inference
# from models import vectors
# from util import chroma

import os
dirname = os.path.dirname(__file__)

app = FastAPI()

# vectors.get_vecs()

# print('Checking if docs have been cached...')
# try:
#     docs = chroma.client.get_collection(name="docs")
#     count = docs.count()
#     print(f"Docs collection already exists, skipping caching. (doc count: {count})")
# except Exception:
#     print('Docs not cached. Storing vectors now.')
#     cache_docs()
#     print('> Done')

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
    print(prediction)
    return {
        'dish_name': prediction
    }

# @app.get("/results", response_class=HTMLResponse)
# async def root(request: Request, query: str):
#     try:
#         results = inference.search(query)
#         for result in results:
#             result['summary'] = result['doc_text'].replace("\\r\\n", "")[0:200]
#         return templates.TemplateResponse(
#             request=request, name="index.j2", context={"query": query, "results": results}
#         )
#     except Exception:
#         logging.exception("Query failed")
#         return templates.TemplateResponse(
#             request=request, name="index.j2", context={"query": query, "error": "Something went wrong"}
#         )


# @app.get("/lucky", response_class=RedirectResponse)
# async def root(request: Request):
#     query = inference.get_random_query()

#     return RedirectResponse(url=f"/results?query={query}", status_code=302)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
