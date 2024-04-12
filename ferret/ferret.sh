#!/bin/bash

RUN=./emp-ot/run
EXE=./emp-ot/bin/test_ferret
LOGOT=24

mkdir -p data/ ../results/

$RUN $EXE $LOGOT

for NGPU in 1 2 4 8
do

for LOGOT in {22..25}
do
    $RUN $EXE $LOGOT $NGPU
done

done

# ulimit -n 1024
# valgrind --leak-check=full $RUN $EXE $LOGOT

# compute-sanitizer --tool memcheck --leak-check full --target-processes all $RUN $EXE $LOGOT
# nsys profile --stats=true $RUN $EXE $LOGOT
