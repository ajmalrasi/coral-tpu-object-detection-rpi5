# Same base image as the server
FROM arm64v8/python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt 

COPY *.py ./

# Command to start your client script
CMD ["python", "client.py"]