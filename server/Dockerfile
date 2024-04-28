FROM arm64v8/python:3.9-slim-buster

ARG USER=nonroot
ARG UID=1000

WORKDIR /app

COPY --chown=$UID ./requirements.txt ./

RUN apt-get update && useradd -m -U -u $UID $USER && apt-get install -y gdal-dev

RUN pip install -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/ pycoral

COPY --chown=$UID *.py ./

EXPOSE 8000

# Command to start the server 
CMD ["uvicorn", "detection_server:app", "--host", "0.0.0.0", "--port", "8000"]