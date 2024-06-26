import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import io
import platform
import time


is_raspberry_pi = platform.system() == 'Linux' and platform.machine().startswith('aarch64')
if is_raspberry_pi:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from utils import init_model, read_yaml, detection, set_inputs

    interpreter, labels = init_model()
    config = read_yaml()
    

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
            start = time.perf_counter()
            image, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.LANCZOS))
            if config["model"]["type"] == "SSD":

                interpreter.invoke()
                inference_time = time.perf_counter() - start
                objs = detect.get_objects(interpreter, config['output']['postprocessing']["confidence_threshold"], scale)
                print('%.2f ms' % (inference_time * 1000))
            elif config["model"]["type"] == "YOLO":

                scaled_tensor = set_inputs(image)
                interpreter.set_tensor(config['input']['tensor_id'], scaled_tensor)
                interpreter.invoke()
                objs = detection(interpreter)
            
            response = {"status": "success", "predictions": []}
            for obj in objs:
                label = labels.get(obj.id, obj.id)
                response["predictions"].append({
                    "label": label,
                    "id": obj.id,
                    "score": obj.score,
                    "bbox": {
                        "xmin": int(obj.bbox.xmin),
                        "ymin": int(obj.bbox.ymin),
                        "xmax": int(obj.bbox.xmax),
                        "ymax": int(obj.bbox.ymax)
                    }
                })
        else:
            response = {"status": "success", "predictions": []}

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)