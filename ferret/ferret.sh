#!/bin/bash

EXE=./ot
LOGOT=24

$EXE $LOGOT

# for LOGOT in 22 23 24 25
# do
#     $EXE $LOGOT
# done

# valgrind --leak-check=full $EXE $LOGOT

# compute-sanitizer --tool memcheck --leak-check full all $EXE $LOGOT
# nsys profile --stats=true $EXE $LOGOT
