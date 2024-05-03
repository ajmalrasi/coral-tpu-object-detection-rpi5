FROM arm64v8/python:3.9-slim-bullseye AS build

ARG USER=nonroot
ARG UID=1000

WORKDIR /app

USER root
RUN apt-get update && useradd -m -U -u $UID $USER && apt-get install -y --no-install-recommends libgdal-dev build-essential wget && \
    rm -rf /var/lib/apt/lists/* 

USER $UID:$UID
COPY --chown=$UID requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://google-coral.github.io/py-repo/ pycoral

# COPY scripts/install_openssl.sh .
# USER root
# RUN chmod +x ./install_openssl.sh && ./install_openssl.sh

# Final Stage
FROM arm64v8/python:3.9-slim-bullseye

ARG USER=nonroot
ARG UID=1000

USER root
RUN useradd -m -U -u $UID $USER
WORKDIR /app
# COPY --from=build /usr/lib/aarch64-linux-gnu/libcrypto.so.1.1 /usr/lib/aarch64-linux-gnu/
# COPY --from=build /usr/lib/aarch64-linux-gnu/libssl.so.1.1 /usr/lib/aarch64-linux-gnu/

USER $UID:$UID

COPY --from=build --chown=$UID /app/ /app/
COPY --from=build --chown=$UID /home/nonroot/.local/ /home/nonroot/.local/
COPY --chown=$UID *.py .
COPY --chown=$UID *.py /models/ ./models/

EXPOSE 8000
CMD ["/home/nonroot/.local/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]