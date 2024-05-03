#!/bin/bash

OPENSSL_VERSION="1.1.1o"

mkdir -p /home/nonroot/opt /home/nonroot/opt/lib
cd /home/nonroot/opt
wget https://www.openssl.org/source/openssl-${OPENSSL_VERSION}.tar.gz
tar -zxvf openssl-${OPENSSL_VERSION}.tar.gz
cd openssl-${OPENSSL_VERSION}

./config && \
make && \
mv libcrypto.so.1.1 libssl.so.1.1 /usr/lib/aarch64-linux-gnu/