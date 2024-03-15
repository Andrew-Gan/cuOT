#!/bin/bash

EXE=./ot
LOGOT=24
TREE=8

$EXE 1 $LOGOT $TREE 1000

# for LOGOT in {22..25}
# do
#     $EXE 1 $LOGOT $TREE 1000
# done

# for NUMTREE in 2 4 $TREE 16 32 64
# do
#     $EXE 1 $LOGOT $NUMTREE 1000
# done

# for BW in 1 10 100 1000
# do
#     $EXE 1 $LOGOT $TREE $BW
# done

# ulimit -n 1024
# valgrind $EXE 1 $LOGOT $TREE 1000
# compute-sanitizer --tool memcheck --leak-check full $EXE 1 $LOGOT $TREE 1000
# nsys profile --stats=true $EXE 1 $LOGOT $TREE 1000
