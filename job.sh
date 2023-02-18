#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

make

# for size in {15..25}
# do
#     for threads in {1..8}
#     do
#         ./aes testData/input.txt testData/key.txt $((2**$size)) $(($threads)) >> out
#     done
# done

for depth in {2..25}
do
    ./aes exp $depth >> out
done

nsys profile --stats=true --output=nsys-stats ./aes exp 20 > out-nsys

make clean
