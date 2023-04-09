#include "gemm_gpu.h"

__host__
void mult_recver_gpu(Matrix ldpc, int *nonZeroRows, int numTrees) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Matrix *d_ldpc;
    Matrix tmp;
    int ldpcByteCols = (ldpc.cols - 1) / 8 + 1;
    int ldpcSize = ldpc.rows * ldpcByteCols;
    cudaMemcpy(&tmp, &ldpc, sizeof(ldpc), cudaMemcpyHostToHost);
    cudaMalloc(&tmp.data, ldpcSize);
    cudaMemcpy(tmp.data, ldpc.data, ldpcSize, cudaMemcpyHostToDevice);
    cudaMalloc(&d_ldpc, sizeof(ldpc));
    cudaMemcpy(d_ldpc, &tmp, sizeof(ldpc), cudaMemcpyHostToDevice);

    int *d_nonZeroRows;
    cudaMalloc(&d_nonZeroRows, numTrees * sizeof(*d_nonZeroRows));
    cudaMemcpy(d_nonZeroRows, nonZeroRows, numTrees * sizeof(*nonZeroRows), cudaMemcpyHostToDevice);

    uint8_t *d_randomVec;
    int randVecByteCount = (ldpc.rows - 1) / 8 + 1;
    cudaMalloc(&d_randomVec, randVecByteCount * sizeof(*d_randomVec));
    int nBlock = (ldpc.rows - 1) / 8 / 512 + 1;
    gemm_gpu<<<nBlock, 512>>>(d_randomVec, d_ldpc, d_nonZeroRows, numTrees);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);
    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Matrix mult using GPU: %0.4f ms\n", duration / NUM_SAMPLES);
}

__global__
void gemm_gpu(uint8_t *randomVec, Matrix *ldpc, int *nonZeroRows, int numTrees) {
    int rowstart = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    int rowend = rowstart + 7;
    int col = 0, idx = 0;
    uint8_t res = 0, bit = 0;

    for (int row = rowstart; row <= rowend; row++) {
        if (row >= ldpc->rows) {
            continue;
        }
        for (int t = 0; t < numTrees; t++) {
            col = nonZeroRows[t];
            idx = row * ldpc->cols + col;
            bit = (ldpc->data[idx / 8] & (1 << idx % 8)) >> idx % 8;
            res ^= bit;
        }
        randomVec[row / 8] &= ~(1 << row % 8);
        randomVec[row / 8] |= res << (row % 8);
    }
}
