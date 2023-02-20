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

for depth in {8..24}
do
    for thread in 1 2 4 8 16
    do
        ./aes exp $depth $thread >> out
    done
done

nsys profile --stats=true --output=nsys-stats ./aes exp 24 16 > out-nsys

make clean
