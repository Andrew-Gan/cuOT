#!/bin/bash

RUN=./emp-ot/run
EXE=./emp-ot/bin/test_main
LOGOT=24

$RUN $EXE $LOGOT

# for LOGOT in 22 23 24 25
# do
#     $RUN $EXE $LOGOT
# done

# ulimit -n 1024
# valgrind --leak-check=full $RUN $EXE $LOGOT

# compute-sanitizer --tool memcheck --leak-check full --target-processes all $RUN $EXE $LOGOT
# nsys profile --stats=true $RUN $EXE $LOGOT
