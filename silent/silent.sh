#!/bin/bash

EXE=./ot
LOGOT=25
TREE=8
NGPU=8

mkdir -p ../results/

# $EXE $LOGOT $TREE $NGPU

for LOGOT in {22..25}
do

for NGPU in 1 2 4 8
do
    $EXE $LOGOT $TREE $NGPU
done

done

# for NUMTREE in 2 4 $TREE 16 32 64
# do
#     $EXE $LOGOT $TREE $NGPU
# done

# ulimit -n 1024
# valgrind $EXE $LOGOT $TREE $NGPU
# compute-sanitizer --tool memcheck --leak-check full $EXE $LOGOT $TREE $NGPU
# nsys profile --stats=true $EXE $LOGOT $TREE $NGPU
