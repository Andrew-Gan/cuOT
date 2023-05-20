#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys

make -j

./ot 1 14 4 data/log-14.txt
./ot 1 16 4 data/log-16.txt
./ot 1 18 4 data/log-18.txt
./ot 1 20 4 data/log-20.txt
nsys profile --stats=true --output=nsys-stats ./ot 1 14 4 data/log-14-nsys.txt

make clean
