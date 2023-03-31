#include <vector>
#include "gemm_gpu.h"

__host__
void mult_recver_gpu(Matrix ldpc, TreeNode *d_multiPprf, int *nonZeroRows, int numTrees) {
    Matrix d_ldpc;
    int ldpcByteCols = (ldpc.cols - 1) / 8 + 1;
    int ldpcSize = ldpc.rows * ldpcByteCols;
    cudaMemcpy(&d_ldpc, &ldpc, sizeof(ldpc), cudaMemcpyHostToDevice);
    cudaMalloc(&d_ldpc.data, ldpcSize);
    cudaMemcpy(d_ldpc.data, ldpc.data, ldpcSize, cudaMemcpyHostToDevice);

    // determine non-zero rows of sparse vector
    int *d_nonZeroRows;
    cudaMalloc(&d_nonZeroRows, numTrees * sizeof(*d_nonZeroRows));
    cudaMemcpy(d_nonZeroRows, nonZeroRows, numTrees * sizeof(*nonZeroRows), cudaMemcpyHostToDevice);

    uint8_t *d_randomVec;
    int randVecByteRows = (ldpc.rows - 1) / 8 + 1;
    cudaMalloc(&d_randomVec, randVecByteRows * sizeof(*d_randomVec));
    dim3 nBlock = ldpc.rows / 512;
    gemm_gpu<<<nBlock, 512>>>(d_randomVec, d_ldpc, d_nonZeroRows, numTrees);
    cudaDeviceSynchronize();
}

__global__
void gemm_gpu(uint8_t *randomVec, Matrix ldpc, int *nonZeroRows, int numTrees) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = 0, idx = 0;
    uint8_t res = 0, bit = 0;

    for(int t = 0; t < (numTrees - 1) / 8 + 1; t++) {
        col = nonZeroRows[t];
        idx = row * ldpc.cols + col;
        bit = (ldpc.data[idx / 8] & (1 << idx % 8)) >> idx % 8;
        res ^= bit;
    }

    randomVec[row / 8] &= ~(1 << row % 8);
    randomVec[row / 8] |= res << row % 8;
}
