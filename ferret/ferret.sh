#!/bin/bash

RUN=./emp-ot/run
EXE=./emp-ot/bin/test_ferret
LOGOT=24
NGPU=1

mkdir -p data/ ../results/

$RUN $EXE $LOGOT $NGPU

# for LOGOT in 25
# do

# for NGPU in 1 2 4 8
# do
#     $RUN $EXE $LOGOT $NGPU
# done

# done

# ulimit -n 1024
# valgrind --leak-check=full $RUN $EXE $LOGOT $NGPU

# compute-sanitizer --tool memcheck --leak-check full --target-processes all $RUN $EXE $LOGOT $NGPU
# nsys profile --stats=true $RUN $EXE $LOGOT
