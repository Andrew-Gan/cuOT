#!/bin/bash

docker rm -f gpuot-test
docker build -t gpuot .
docker run -d --name gpuot-test --runtime=nvidia --gpus all -it \
    -m=500m --kernel-memory=500m --memory-swap=500m \
    -v $(pwd)/result:/home/gpuot/result gpuot