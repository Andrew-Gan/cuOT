#!/bin/bash

./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 30
# valgrind ./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 30
# compute-sanitizer --tool memcheck --target-processes all ./ferret/emp-ot/run ./ferret/emp-ot/bin/test_ferret 30
# nsys profile --stats=true ./run ./bin/test_ferret 30

# sbatch -n 4 -N 1 --gpus-per-node=1 -A standby job.sh
