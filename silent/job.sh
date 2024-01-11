#!/bin/bash

cd $SLURM_SUBMIT_DIR
./ot 1 24 8

# for NUMTREE in 2 4 8 16 32 64 128 256
# do
#     ./ot 1 24 $NUMTREE
# done

# for LOGOT in 20 21 22 23 24
# do
#     ./ot 1 $LOGOT 8
# done

# valgrind ./ot 1 24 8
# compute-sanitizer --tool memcheck ./ot 1 24 8
