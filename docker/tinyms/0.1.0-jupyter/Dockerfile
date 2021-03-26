ARG BASE_CONTAINER=jupyter/scipy-notebook:ubuntu-18.04
FROM $BASE_CONTAINER

LABEL MAINTAINER="Leon Wang <wanghui71leon@gmail.com>"

# Set the default jupyter token with "tinyms"
RUN sh -c '/bin/echo -e "tinyms\ntinyms\n" | jupyter notebook password'

# Install TinyMS cpu whl package
RUN pip install --no-cache-dir numpy==1.17.5 tinyms==0.1.0 && \
    fix-permissions "${CONDA_DIR}"
