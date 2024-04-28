FROM arm64v8/python:3.9-slim-buster

ARG USER=nonroot
ARG UID=1000

USER $UID:$UID
WORKDIR /app

COPY --chown=$UID ./requirements.txt ./

USER root

RUN apt-get update && useradd -m -U -u $UID $USER && apt-get install -y libgdal-dev

USER $UID:$UID
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/ pycoral

COPY --chown=$UID *.py ./

EXPOSE 8000

CMD ["/home/$USER/.local/bin/uvicorn", "detection_server:app", "--host", "0.0.0.0", "--port", "8000"]
