# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get -y install build-essential python3 cmake git libssl-dev

COPY gpu /home/gpuot/gpu
WORKDIR /home/gpuot/gpu
RUN make -j -s

COPY ferret /home/gpuot/ferret
WORKDIR /home/gpuot/ferret
RUN git clone https://github.com/emp-toolkit/emp-tool.git
WORKDIR /home/gpuot/ferret/emp-tool
RUN git checkout 44b1dde
RUN cmake -DCMAKE_INSTALL_PREFIX=../lib
RUN make -j4
RUN make install

WORKDIR /home/gpuot/ferret/emp-ot
RUN cmake -DCMAKE_INSTALL_PREFIX=../lib -DCMAKE_C_FLAGS='-O3' -DCMAKE_CUDA_FLAGS='-O3'
RUN make -j4
RUN make install

WORKDIR /home/gpuot/ferret
RUN chmod +x ferret.sh
CMD ["./ferret.sh"]
# CMD ["tail", "-f", "/dev/null"]
