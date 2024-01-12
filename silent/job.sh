#!/bin/bash

cd $SLURM_SUBMIT_DIR
./ot 1 24 8 1000

# for NUMTREE in 2 4 8 16 32 64 128 256
# do
#     ./ot 1 24 $NUMTREE 1000
# done

# for LOGOT in 20 21 22 23 24
# do
#     ./ot 1 $LOGOT 8 1000
# done

# for BW in 1 10 100 1000
# do
#     ./ot 1 24 8 $BW
# done

# valgrind ./ot 1 24 8
# compute-sanitizer --tool memcheck ./ot 1 24 8 1000
# nsys profile --stats=true ./ot 1 24 8 1000
