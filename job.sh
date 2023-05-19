#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys

make -j

./ot 1 14 4 log-14.txt
./ot 1 16 4 log-16.txt
./ot 1 18 4 log-18.txt
./ot 1 20 4 log-20.txt
nsys profile --stats=true --output=nsys-stats ./ot 1 14 4

make clean
