#  Real-Time Object Detection on Raspberry Pi 5 with Coral Edge TPU

This project enables real-time object detection on the Coral Edge TPU integrated with a Raspberry Pi 5. With the FastAPI server, you can easily send images and receive a list of detected objects accompanied by bounding boxes, confidence scores, and labels.

## Key Features

 - Fast and efficient object detection: Leverages the Coral Edge TPU for hardware-accelerated machine learning.
 - Easy-to-use API: Simple image submission and result retrieval.
 - Example usage and visualization: Provides clear examples for using and understanding the results.

## Prerequisites

- Raspberry Pi 5 with Coral Edge TPU M.2 attached to PCIe slot.
- Compatible Operating System:
  - Debian GNU/Linux 12 (bookworm) 64 bit
- Linux kernel 6.6.20+rpt-rpi-v8. Check with `uname -r`.
- Python 3.9.16
- Linux PCIe Driver for Coral Edge TPU M.2 [documentation](https://coral.ai/docs/notes/build-coral/)

### Requirements

To install Docker, follow the instructions in the [official Docker documentation](https://docs.docker.com/engine/install/debian/#install-using-the-repository).


```bash
sudo apt install devscripts debhelper dkms -y

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
```
Install libedgetpu
```bash
sudo apt-get install libedgetpu1-std
```

## Running the Server

Start the FastAPI server:

```bash
docker pull ghcr.io/ajmalrasi/object_detection_tpu:main
```

```bash
docker run --device=/dev/apex_0:/dev/apex_0 -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro -p 8000:8000 -it <image>
```
