#!/bin/bash

mkdir -p results
docker rm -f gpuot-test
docker build -t gpuot . && docker run --name gpuot-test --runtime=nvidia --gpus '"device=0,1"' -it \
    -m=500m --kernel-memory=500m --memory-swap=500m \
    -v $(pwd)/results:/home/gpuot/results gpuot