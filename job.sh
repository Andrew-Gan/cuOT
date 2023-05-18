#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys

make -j

./ot 1 14 1
nsys profile --stats=true --output=nsys-stats ./ot 1 14 1

make clean
