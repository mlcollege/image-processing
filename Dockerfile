FROM ubuntu:16.04
MAINTAINER adam.kolar@ceai.io

USER root
ENV SHELL /bin/bash
EXPOSE 8000 6006

RUN mkdir /notebooks
RUN mkdir /tensorboard_summaries
RUN mkdir /data
COPY ./data /data
#COPY ./notebooks/4b_conv_img_net_complete.ipynb /notebooks
#COPY ./notebooks/utils /notebooks/utils

# install all OS dependencies for fully functional notebook server
RUN apt-get update && apt-get install -yq --no-install-recommends \
    git \
    vim \
    wget \
    build-essential \
    python-dev \
    ca-certificates \
    bzip2 \
    unzip \
    libsm6 \
    sudo \
    locales \
    libzmq3-dev \
    python3-pip \
    python3-dev \
    && apt-get clean

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen
ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'

# install numpy, scipy, atlas
RUN apt-get update && apt-get install -yq --no-install-recommends \
    libatlas-dev \
    python3-numpy \
    python3-scipy \
    python3-pillow \
    && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install --upgrade ipython[notebook] ipywidgets
RUN pip3 install --upgrade matplotlib
RUN pip3 install -r /data/requirements
RUN jupyter nbextension enable --py widgetsnbextension

COPY run.sh /
ENTRYPOINT ./run.sh
