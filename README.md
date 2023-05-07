# GPU Accelerated OT Protocols

A platform for running existing or designing new Oblivious Transfer protocols by harnessing the power of hardware acceleration on the GPU.




## Software tools:
* gcc 9.3.0
* nvcc 11.2

## Dependencies:
**Ubuntu/Debian**:
```
apt-get install gcc nvcc
```

## Run Instruction:
```
make
./pprf <protocol id> <depth or logOT> <number of trees>
```

## Protocols Currently Available

0. OT
1. Silent OT

[1] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, and P. Scholl, ‘Efficient Pseudorandom Correlation Generators: Silent OT Extension and More’, in Advances in Cryptology – CRYPTO 2019, A. Boldyreva and D. Micciancio, Eds., in Lecture Notes in Computer Science, vol. 11694. Cham: Springer International Publishing, 2019, pp. 489–518. doi: 10.1007/978-3-030-26954-8_16.

[2] E. Boyle et al., ‘Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation’, in Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, London United Kingdom: ACM, Nov. 2019, pp. 291–308. doi: 10.1145/3319535.3354255.

