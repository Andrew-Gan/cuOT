#!/bin/bash

RUN=./emp-ot/run
EXE=./emp-ot/bin/test_ferret
LOGOT=24

mkdir -p data/

# $RUN $EXE $LOGOT

for LOGOT in 22 23 24 25
do
    $RUN $EXE $LOGOT
    rm data/*
done

# ulimit -n 1024
# valgrind --leak-check=full $RUN $EXE $LOGOT

# compute-sanitizer --tool memcheck --leak-check full --target-processes all $RUN $EXE $LOGOT
# nsys profile --stats=true $RUN $EXE $LOGOT
