# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential libsodium-dev

WORKDIR /home/gpuot
RUN mkdir result

COPY gpu gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

WORKDIR /home/gpuot
COPY silent silent
WORKDIR /home/gpuot/silent
RUN make -j -s

CMD ["tail", "-f", "/dev/null"]
# CMD ["./ot", "1", "24", "8", "1000"]
