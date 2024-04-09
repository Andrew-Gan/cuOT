#!/bin/bash

EXE=./ot
LOGOT=24
TREE=8

$EXE 1 $LOGOT $TREE

# for LOGOT in {22..25}
# do
#     $EXE 1 $LOGOT $TREE
# done

# for NUMTREE in 2 4 $TREE 16 32 64
# do
#     $EXE 1 $LOGOT $NUMTREE
# done

# for BW in 1 10 100
# do
#     $EXE 1 $LOGOT $TREE $BW
# done

# ulimit -n 1024
# valgrind $EXE 1 $LOGOT $TREE
# compute-sanitizer --tool memcheck --leak-check full $EXE 1 $LOGOT $TREE
# nsys profile --stats=true $EXE 1 $LOGOT $TREE
