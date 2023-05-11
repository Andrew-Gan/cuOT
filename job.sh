#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys

make -j 4

./ot 1 11 75 out-11.txt
./ot 1 14 73 out-14.txt
./ot 1 17 72 out-17.txt
./ot 1 20 70 out-20.txt
./ot 1 23 69 out-23.txt

make clean
