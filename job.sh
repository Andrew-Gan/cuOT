#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys slurm*

make

for depth in 10 12 14
do
    for tree in 5 11
    do
        ./pprf $depth $tree >> out
    done
done

nsys profile --stats=true --output=nsys-stats ./pprf 24 32 > out-nsys

make clean
