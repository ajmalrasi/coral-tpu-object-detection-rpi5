FROM arm64v8/python:3.9-slim-buster AS build

ARG USER=nonroot
ARG UID=1000

WORKDIR /app

USER root
RUN apt-get update && useradd -m -U -u $UID $USER && apt-get install -y --no-install-recommends libgdal-dev && \
    rm -rf /var/lib/apt/lists/* 


USER $UID:$UID
COPY --chown=$UID requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# Final Stage
FROM arm64v8/python:3.9-slim-buster

ARG USER=nonroot
ARG UID=1000

USER root
RUN useradd -m -U -u $UID $USER
RUN apt-get update && apt-get install -y --no-install-recommends libssl-dev 

USER $UID:$UID
WORKDIR /app

COPY --from=build --chown=$UID /app/ /app/
COPY --from=build --chown=$UID /home/nonroot/.local/ /home/nonroot/.local/
COPY --chown=$UID *.py .
COPY --chown=$UID *.py /models/ ./models/


EXPOSE 8000
CMD ["/home/nonroot/.local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]