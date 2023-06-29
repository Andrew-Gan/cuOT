#!/bin/bash

cd $SLURM_SUBMIT_DIR

make -j -s
mkdir -p output

# for NUMTREE in 2 4 8 16 32 64
# do
#     ./ot 1 24 $NUMTREE
# done

# for LOGOT in 20 21 22 23 24
# do
#     ./ot 1 $LOGOT 8
# done

./ot 1 24 8
# compute-sanitizer --tool memcheck ./ot 1 24 8 &> out

python plotter.py

# nsys profile --stats=true --output=nsys-stats ./ot 1 14 4 data/log-14-nsys.txt

