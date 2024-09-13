FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

RUN apt update
RUN DEBIAN_FRONTEND="noninteractive" apt install libopencv-dev software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa && apt install python3.9 python3-pip python3.9-dev -y
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3
RUN pip3 install --upgrade pip && pip3 install torch==2.0.1 ninja setuptools==68.0.0

WORKDIR /root
ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
WORKDIR /workspace