# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential python3 cmake git libssl-dev valgrind

WORKDIR /home/gpuot
RUN mkdir result

COPY gpu gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

WORKDIR /home/gpuot
COPY ferret ferret
WORKDIR /home/gpuot/ferret
RUN python3 build.py --tool --ot
RUN chmod +x ferret.sh

CMD ["tail", "-f", "/dev/null"]
# CMD ["./ferret.sh"]
