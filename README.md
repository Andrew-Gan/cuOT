# cuOT: Accelerating Oblivious Transfer on GPUs

This repository contains the code for cuOT, which implements OT variants based on recent SilentOT extension constructions on GPUs.

## Hardware Requirements:
* GPU CUDA Compute Capability >= 7.0 (V100 or newer)

## Software Dependencies:
We provide a docker envoronment to build and run the library. To do so, you need docker installed and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) set up to build and run our GPU containers.

To build and run without docker, you need the following libraries:
* gcc >= 12.3.0
* nvcc >= 12.1  

## Build and run instructions on Docker (recommended):
Build:
```
docker compose build
```
Run SOT and Ferret:
```
docker compose up silent
docker compose up ferret
```

To set the number of OTs for each run and GPUs to use, change `LOGOT` and `NGPU` parameters in `silent/silent.sh` and `ferret/ferret.sh`. 

## Build and run instructions without Docker:
Build and run SOT:
```
cd gpu
make -j
cd ../silent
chmod +x silent.sh
make -j
./silent.sh
```

Build and run Ferret:
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

silent : steps to be conducted by sender and receiver for SOT
├── silent.sh : script to run SOT across input sizes and number of GPUs
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

## Silent OT Constructions Implemented in cuOT

1. [SOT](https://github.com/osu-crypto/libOTe)
2. [Ferret](https://github.com/emp-toolkit/emp-ot)

[1] E. Boyle, G. Couteau, N. Gilboa, Y. Ishai, L. Kohl, and P. Scholl, ‘Efficient Pseudorandom Correlation Generators: Silent OT Extension and More’, in Advances in Cryptology – CRYPTO 2019.

[2] K. Yang, C. Weng, X. Lan, J. Zhang, and X. Wang, 'Ferret: Fast extension for correlated OT with small communication', in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security - CCS 2020.
