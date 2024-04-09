# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential python3 cmake git libssl-dev

WORKDIR /home/gpuot/ferret
COPY ferret/build.py .
RUN python3 build.py --tool

COPY gpu /home/gpuot/gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

COPY ferret /home/gpuot/ferret
WORKDIR /home/gpuot/ferret
RUN python3 build.py --ot
RUN chmod +x ferret.sh

# WORKDIR emp-ot/bin
# CMD ["tail", "-f", "/dev/null"]
CMD ["./ferret.sh"]
