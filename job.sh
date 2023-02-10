#!/bin/bash

module unload intel
module load gcc

cd $SLURM_SUBMIT_DIR

rm -f out
make

for size in {15..25}
do
    for threads in {1..8}
    do
        ./aes testData/input.txt testData/key.txt $((2**$size)) $(($threads)) >> out
    done
done

make clean
