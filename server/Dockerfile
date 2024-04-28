FROM arm64v8/python:3.9-slim-buster  

WORKDIR /app

COPY ./requirements.txt ./

RUN sudo apt-get update && sudo apt-get install -y gdal-dev

RUN pip install -r requirements.txt

COPY *.py ./

EXPOSE 8000

# Command to start the server 
CMD ["uvicorn", "detection_server:app", "--host", "0.0.0.0", "--port", "8000"]