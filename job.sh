#!/bin/bash

module unload intel
module load gcc/9.3.0

cd $SLURM_SUBMIT_DIR

rm -f nsys* out out-nsys slurm*

make

# for depth in 14 17 20
# do
#     for tree in 5 11
#     do
#         ./pprf $depth $tree >> out
#     done
# done

array=( 14 17 20 )
array2=( 73 72 70 )

for i in "${!array[@]}"; do
    ./pprf "${array[i]}" "${array2[i]}" >> out
done

nsys profile --stats=true --output=nsys-stats ./pprf 14 11 > out-nsys

make clean
