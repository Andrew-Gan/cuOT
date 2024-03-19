#!/bin/bash

docker rm -f memcpy-test
docker build -t memcpy . && docker run --name memcpy-test -d --runtime=nvidia --gpus '"device=0,1"' -it \
    -v $(pwd):/home/memcpy memcpy
