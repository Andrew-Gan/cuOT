#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys

make

./pprf 1 11 75 >> out
./pprf 1 14 73 >> out
./pprf 1 17 72 >> out
./pprf 1 20 70 >> out
./pprf 1 23 69 >> out

nsys profile --stats=true --output=nsys-stats ./pprf 1 14 73 > out-nsys

make clean
