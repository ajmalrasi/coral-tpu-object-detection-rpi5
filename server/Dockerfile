FROM arm64v8/python:3.9-alpine AS build

ARG USER=nonroot
ARG UID=1000

USER root
WORKDIR /app

COPY --chown=$UID requirements.txt ./  
RUN apk add --no-cache --update build-base libgdal-dev \ 
    && useradd -m -U -u $UID $USER \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# Final Stage
FROM arm64v8/python:3.9-alpine

ARG USER=nonroot
ARG UID=1000

USER $UID:$UID
WORKDIR /app

COPY --from=build --chown=$UID /app/ /app/
COPY --chown=$UID *.py ./
COPY --chown=$UID ./models/* models/

EXPOSE 8000
CMD ["/home/nonroot/.local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]