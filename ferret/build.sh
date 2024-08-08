#!/bin/bash

target=emp-tool
if [ ! -d $target ]; then
    git clone https://github.com/emp-toolkit/emp-tool.git
fi
cd $target
git checkout 44b1dde
cmake -DCMAKE_INSTALL_PREFIX=../lib \
    -DCMAKE_C_FLAGS='-g' -DCMAKE_CUDA_FLAGS='-g -G'
make -j4
make install
cd ..

target=emp-ot
cd $target
cmake -DCMAKE_INSTALL_PREFIX=../lib \
    -DCMAKE_C_FLAGS='-g' -DCMAKE_CUDA_FLAGS='-g -G'
make -j4
make install
cd ..
