#include "rand_gpu.h"
#include <curand_kernel.h>

Matrix gen_rand_gpu(size_t height, size_t width) {
    Matrix d_randMatrix = { .rows = height, .cols = width };
    cudaMalloc(&d_randMatrix.data, height * width / 8);

    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(prng, clock());
    curandGenerateUniform(prng, (float*) d_randMatrix.data, width * height / 32);

    return d_randMatrix;
}

Matrix gen_ldpc_gpu(int numLeaves, int numTrees) {
    Matrix ldpc;
    ldpc.cols = numLeaves * sizeof(TreeNode) * 8;
    ldpc.rows = ldpc.cols / 2;
    ldpc.data = (uint8_t*) calloc(ldpc.rows * numLeaves, 1);

    // generate left
    int gap = ldpc.rows / numTrees;
    for (int i = 0; i < numTrees; i++) {
        int row = i * gap;
        int col = 0;
        while (row < ldpc.rows && col < ldpc.cols / 2) {
            int idx = row++ * ldpc.cols + col++;
            ldpc.data[idx / 8] |= 1 << (idx % 8);
        }
    }
    for (int i = 1; i < numTrees; i++) {
        int r = 0;
        int c = i * gap;
        while (r < ldpc.rows && c < ldpc.cols / 2) {
            int idx = r++ * ldpc.cols + c++;
            ldpc.data[idx / 8] |= 1 << (idx % 8);
        }
    }

    // generate right
    uint8_t sampled_rows[4] = {0x42, 0x49, 0xea, 0xde};
    for (int r = 0; r < ldpc.rows; r++) {
        int c = ldpc.cols / 2 + r;
        int idx = r * ldpc.cols + c;
        ldpc.data[idx / 8] = sampled_rows[idx % 4];
    }

    return ldpc;
}
