import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
import os
inf = os.walk('.')
print([x[0] for x in inf])

import uvicorn
import platform

app = FastAPI()

print(platform.system())
print(platform.machine())
is_raspberry_pi = platform.system() == 'Linux' and platform.machine().startswith('arm')
if is_raspberry_pi:
    print("Raspberry pi")
    from .utils import load_model
    load_model()

@app.post("/predict")
async def process_image(data: dict = Body(...)):
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request")

    try:
        image_str = data["image"]
        decoded_image = base64.b64decode(image_str)

        return JSONResponse(content={"message": "Image received and processed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)