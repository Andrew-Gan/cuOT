#!/bin/bash

cd $SLURM_SUBMIT_DIR

EXE=./emp-ot/bin/test_ferret
LOGOT=24

# ./emp-ot/run $EXE $LOGOT

for LOGOT in 22 23 24 25
do
    ./emp-ot/run $EXE $LOGOT
done

# valgrind --leak-check=full ./ferret/emp-ot/run $EXE $LOGOT

# compute-sanitizer --tool memcheck --target-processes all $EXE 1 12345 $LOGOT & compute-sanitizer --tool memcheck --target-processes all $EXE 2 12345 ${@:2} $LOGOT
# nsys profile --stats=true ./ferret/emp-ot/run $EXE $LOGOT
