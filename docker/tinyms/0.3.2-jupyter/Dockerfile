ARG BASE_CONTAINER=jupyter/scipy-notebook:ubuntu-18.04
FROM $BASE_CONTAINER

LABEL MAINTAINER="TinyMS Authors"

# Set the default jupyter token with "tinyms"
RUN sh -c '/bin/echo -e "tinyms\ntinyms\n" | jupyter notebook password'

# Install TinyMS cpu whl package
RUN pip install --upgrade -i https://pypi.mirrors.ustc.edu.cn/simple pip && \
    pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple numpy==1.17.5 tinyms==0.3.2 && \
    pip install --upgrade jupyter notebook jupyterhub nbconvert -i https://pypi.mirrors.ustc.edu.cn/simple && \
    fix-permissions "${CONDA_DIR}"
