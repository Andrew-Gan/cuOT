#!/bin/bash

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys
make -j -s
mkdir -p data

for NUMTREE in 2 4 8 16 32 64
do
    ./ot 1 20 $NUMTREE data/log-20-$NUMTREE.txt
done

python plotter.py

# nsys profile --stats=true --output=nsys-stats ./ot 1 14 4 data/log-14-nsys.txt

make clean
