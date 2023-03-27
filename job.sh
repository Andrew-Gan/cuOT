#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys slurm*

make

# for size in {15..25}
# do
#     for threads in {1..8}
#     do
#         ./aes testData/input.txt testData/key.txt $((2**$size)) $(($threads)) >> out
#     done
# done

for thread in 2 4 8 16
do
    for depth in {12..24}
    do
        ./aes exp $depth $thread >> out
    done
done

nsys profile --stats=true --output=nsys-stats ./aes exp 24 16 > out-nsys

make clean
