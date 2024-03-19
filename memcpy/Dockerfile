# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential python3 pip
RUN pip install matplotlib

WORKDIR /home

COPY . memcpy
WORKDIR /home/memcpy

RUN nvcc memcpy.cu -o memcpy

CMD ["tail", "-f", "/dev/null"]
