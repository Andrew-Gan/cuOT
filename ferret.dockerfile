# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential python3 cmake git libssl-dev

COPY gpu /home/gpuot/gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

COPY ferret /home/gpuot/ferret
WORKDIR /home/gpuot/ferret
RUN chmod +x build.sh
RUN bash build.sh
RUN chmod +x ferret.sh

CMD ["./ferret.sh"]
# CMD ["tail", "-f", "/dev/null"]