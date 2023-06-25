#!/bin/bash

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys
make -j -s
mkdir -p output

for NUMTREE in 8
do
    ./ot 1 20 $NUMTREE
done

python plotter.py

# nsys profile --stats=true --output=nsys-stats ./ot 1 14 4 data/log-14-nsys.txt

make clean
