# Object Detection on Coral Edge TPU with PyCoral

This project allows you to perform real-time object detection on the Coral Edge TPU using a FastAPI server. You can send images to the server, and it will return a list of detected objects along with their bounding boxes, scores, and labels.

## Key Features

- Fast and efficient object detection on the Edge TPU
- Easy-to-use API for image submission and result retrieval
- Example usage and visualization code

## Prerequisites

- Raspberry Pi 5 with Coral Edge TPU M.2 attached to PCie
- Compatible Operating System:
  - Debian GNU/Linux 12 (bookworm) 64 bit
- Linux kernel 6.6.20+rpt-rpi-v8. Check with uname -r.
- Python 3.9.16

### Required Python packages
- tflite-runtime 2.0.0
- PyCoral 2.5.0


## Running the Server

Start the FastAPI server:

```bash
docker run --device=/dev/apex_0:/dev/apex_0 -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro -p 8000:8000 -it <image>