# GPU Accelerated OT Protocols

This repository contains the code for cuOT, which implements OT variants based on recent SilentOT extension constructions on GPUs.

## Hardware Requirements:
* GPU CUDA Compute Capability >= 7.0 (V100 or newer)

## Software Dependencies:
### For execution with docker
* Docker >= 26.1.0
* [Nvidia container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)  
### For execution without docker
* gcc >= 12.3.0
* nvcc >= 12.1  
* cmake

## Setting number of OTs or GPUs to use:
1. Go into silent.sh or ferret.sh
2. Find the line that says `LOGOT=XX` or `NGPU=X`
3. Changing the argument to desired values

Note:  
i. LOGOT should take on values no greater than 28 to ensure memory sufficiency. 
ii. NGPU should take on a multiple of 2 and not exceed number of GPUs available.  

## Run instructions on Docker (recommended):
To build everything
```
docker compose build
```
To run Silent OT
```
docker compose up silent
```

To run Ferret
```
docker compose up ferret
```

## Run instructions without Docker:
To run Silent OT
```
cd gpu
make -j
cd ../silent
chmod +x silent.sh
make -j
./silent.sh
```

To run Ferret
```
cd gpu
make -j
cd ../ferret
build.py --tool --ot
chmod +x ferret.sh
./ferret.sh
```

## Directory structure:
```
gpu : includes operations for AES, BaseOT, data structure manipulation
├── Makefile : instructions for execution
├── aes_xxx : AES related operations
├── gpu_xxx : GPU data strcuture permutations
├── base classes : for inheritance into tree expansion and LPN

silent : steps to be conducted by sender and receiver for Silent OT
├── silent.sh : script to run Silent OT across input sizes and number of GPUs
├── Makefile: instructions for execution
├── lib : cryptoTools, libdivide and libsodium for hashing, PRNGs, ECC etc.
├── sender.cu : orchestrate steps performed by sender
├── recver.cu : orchestrate steps performed by receiver

ferret : steps to be conducted by sender and receiver for Ferret
├── ferret.sh : script to run Ferret across input sizes and number of GPUs
├── build.py : install emp-tool, a prerequisite for emp-ot
├── Makefile: instructions for execution, dependent on build.py
├── lib : linkable object files for emp-ot
├── emp-ot
    ├── ferret : codebase where substitutions in code with GPU impl. were made
        ├── dev_layer.cu : where all CUDA code reside
    ├── test : starting point of execution and CLI parsing for Ferret
```

## Protocols Implemented

1. [Silent OT](https://github.com/osu-crypto/libOTe)
2. [Ferret](https://github.com/emp-toolkit/emp-ot)

[1] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, and P. Scholl, ‘Efficient Pseudorandom Correlation Generators: Silent OT Extension and More’, in Advances in Cryptology – CRYPTO 2019, A. Boldyreva and D. Micciancio, Eds., in Lecture Notes in Computer Science, vol. 11694. Cham: Springer International Publishing, 2019, pp. 489–518. doi: 10.1007/978-3-030-26954-8_16.

[2] E. Boyle et al., ‘Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation’, in Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security, London United Kingdom: ACM, Nov. 2019, pp. 291–308. doi: 10.1145/3319535.3354255.
