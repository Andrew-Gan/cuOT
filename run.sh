#!/bin/bash

mkdir -p results
docker rm -f gpuot-test
docker build -t gpuot . && docker run --name gpuot-test --runtime=nvidia \
    --gpus '"device=0,1"' -m=1g -v $(pwd)/results:/home/gpuot/results gpuot