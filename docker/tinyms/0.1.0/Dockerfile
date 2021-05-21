ARG BASE_CONTAINER=mindspore/mindspore-cpu:1.1.1
FROM $BASE_CONTAINER

LABEL MAINTAINER="Leon Wang <wanghui71leon@gmail.com>"

# Install base tools
RUN apt-get update

# Install TinyMS cpu whl package
RUN pip install --no-cache-dir tinyms==0.1.0
