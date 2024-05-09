FROM arm64v8/python:3.9-slim-buster

ARG USER=nonroot
ARG UID=1000

USER $UID:$UID
WORKDIR /app

COPY --chown=$UID requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=$UID *.py ./

CMD ["python", "client.py"]