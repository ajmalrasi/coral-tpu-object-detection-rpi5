# Object Detection on Coral Edge TPU with FastAPI

This project allows you to perform real-time object detection on the Coral Edge TPU using a FastAPI server. You can send images to the server, and it will return a list of detected objects along with their bounding boxes, scores, and labels.

## Key Features

- Fast and efficient object detection on the Edge TPU
- Easy-to-use API for image submission and result retrieval
- Example usage and visualization code

## Prerequisites

- A Coral Edge TPU device (USB Accelerator or Dev Board)
- Compatible Operating System:
  - Raspberry Pi OS (for Raspberry Pi with the Edge TPU)
  - A Linux system with the Edge TPU runtime installed (for other compatible boards or accelerator)
- Python 3
- Required libraries: `Bash`, `pip install fastapi uvicorn pillow pycoral`

**Use code with caution.**

**Note:** You'll also need to have the TensorFlow Lite runtime and Edge TPU compatible object detection model. See "Setup" below for instructions.

## Setup

1. Install the Edge TPU libraries: Follow the official Coral setup instructions for your device: [Coral Setup](https://coral.ai/docs/setup/)
2. Obtain a compatible object detection model: Download or train a TensorFlow Lite model that has been quantized for the Edge TPU. See available models here: [Available Models](https://coral.ai/models/)
3. Place the model file (e.g., model.tflite) in your project directory

**Modifications to server.py**

- Update the `load_model` function in `server.py` to load your specific Edge TPU compatible TensorFlow Lite model.
- Update the `labels` variable to correspond to the labels of your model.

## Running the Server

Start the FastAPI server:

```bash
uvicorn server:app --reload