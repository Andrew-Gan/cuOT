#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

# rm -f nsys* out out-nsys

make

./pprf 11 75 >> out
./pprf 14 73 >> out
./pprf 17 72 >> out
./pprf 20 70 >> out
./pprf 23 69 >> out
./pprf 26 67 >> out

nsys profile --stats=true --output=nsys-stats ./pprf 14 73 > out-nsys

make clean
