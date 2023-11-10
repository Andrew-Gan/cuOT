#!/bin/bash

./run ./bin/test_ferret 30
# compute-sanitizer --tool memcheck ./run ./bin/test_ferret 24
# nsys profile --stats=true ./run ./bin/test_ferret 24

# sbatch -n 4 -N 1 --gpus-per-node=1 -A standby job.sh
