# GPU Accelerated OT Protocols

Running SilentOT on the GPU.

## Dependencies:
* gcc 12.3.0
* nvcc 12.1

## Run instruction:
```
make
./pprf <protocol id> <depth or logOT> <number of trees>
```

## File structure:
* App Level - integration of low-level modules to perform OT protocols, i.e. Silent OT
* Module Level - independent components that interface with CUDA, i.e. AES, BaseOT
* Device Level - CUDA code to perform GPU operations, i.e. GPU tree exp, matmul

## Protocols Currently Available

0. OT
1. Silent OT
2. Ferret

[1] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, and P. Scholl, ‘Efficient Pseudorandom Correlation Generators: Silent OT Extension and More’, in Advances in Cryptology – CRYPTO 2019, A. Boldyreva and D. Micciancio, Eds., in Lecture Notes in Computer Science, vol. 11694. Cham: Springer International Publishing, 2019, pp. 489–518. doi: 10.1007/978-3-030-26954-8_16.

[2] E. Boyle et al., ‘Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation’, in Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, London United Kingdom: ACM, Nov. 2019, pp. 291–308. doi: 10.1145/3319535.3354255.

