# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential libsodium-dev

COPY gpu /home/gpuot/gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

COPY silent /home/gpuot/silent
WORKDIR /home/gpuot/silent
RUN chmod +x silent.sh
RUN make -j -s

CMD ["./silent.sh"]
