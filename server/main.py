import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import io
import platform

is_raspberry_pi = platform.system() == 'Linux' and platform.machine().startswith('aarch64')
if is_raspberry_pi:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from utils import load_model
    interpreter = load_model()

app = FastAPI()

@app.post("/predict")
async def process_image(data: dict = Body(...)):
    if "image" not in data:
        raise HTTPException(status_code=400, detail="Missing 'image' field in request")

    try:
        image_str = data["image"]
        decoded_image = base64.b64decode(image_str)
        image_bytes = io.BytesIO(decoded_image)
        image = Image.open(image_bytes)
        if is_raspberry_pi:
            _, scale = common.set_resized_input(
                interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))
        print(scale)
        return JSONResponse(content={"message": "Image received and processed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)