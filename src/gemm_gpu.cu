#include "gemm_gpu.h"

/************************************************************
Algorithm generate chunks of full matrix and pass into kernel

Tier    | Dimension (bits)  | Size
Matrix  | 2^20 x 2^21       | 256 GB
Chunk   | 2^15 x 2^15       | 128 MB
Tile    | 2^7  x 2^12       |  64 KB
*no shared mem needed for tile

1 full matrix   = 32x64 chunks
1 chunk         = 32x64 tblocks
1 tile / block  = 32 warps
1 warp          = 64 half-rows in block to add
************************************************************/

#define CHUNK_SIDE 32768

__global__
void mat_vec_mult(Vector out, uint8_t *subTotal, Matrix matrix, Vector vec, int chunkStartCol) {
    // treat the unirand mat as transposed
    // uniform rand mat ~ transposed
    // accessing by row in transposed = accessing by col in original
    // threads in same warp access same row for coalescing

    // grid(8, 256), block(512)
    // mat_vec_mult<<<grid, block>>>

    int startRow = blockIdx.y * 128;
    int col_byte = blockIdx.x * blockDim.x + threadIdx.x;

    for (int row = startRow; row < startRow + 128; row++) {
        if (vec.data[row / 8] & (1 << (row % 8)) != 0) {
            subTotal[blockIdx.y * CHUNK_SIDE + col_byte]
                ^= matrix.data[row * (matrix.cols / 8) + col_byte];
        }
    }

    // sub subtotal into out
    if (blockIdx.y > 0) {
        return;
    }
    for(int i = 0; i < 256; i++) {
        out.data[chunkStartCol + col_byte] ^= subTotal[i * CHUNK_SIDE + col_byte];
    }
}

__host__
void mult_sender_gpu(Matrix d_randMatrix, Vector d_fullVec) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint8_t *d_subTotal;
    cudaMalloc(&d_subTotal, 16 * CHUNK_SIDE);
    Vector d_randomVec = { .n = d_randMatrix.rows };
    cudaMalloc(&d_randomVec.data, d_randomVec.n / 8);

    dim3 grid(8, 256);
    dim3 block(512);
    for (int chunkR = 0; chunkR < d_randMatrix.rows / CHUNK_SIDE; chunkR++) {
        for (int chunkC = 0; chunkC < d_randMatrix.cols / CHUNK_SIDE; chunkC++) {
            mat_vec_mult<<<grid, block>>>(d_randomVec, d_subTotal, d_randMatrix,
                d_fullVec, chunkC * CHUNK_SIDE);
            cudaDeviceSynchronize();
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Matrix mult sender using GPU: %0.4f ms\n", duration / NUM_SAMPLES);
}

__host__
void mult_recver_gpu(Matrix d_randMatrix, Vector d_choiceVec, Vector d_puncturedVec) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    uint8_t *d_subTotalChoice, *d_subTotalPunctured;
    cudaMalloc(&d_subTotalChoice, 16 * CHUNK_SIDE);
    cudaMalloc(&d_subTotalPunctured, 16 * CHUNK_SIDE);
    Vector d_choiceVecRand, d_puncturedVecRand;
    cudaMalloc(&d_choiceVecRand.data, d_randMatrix.rows / 8);
    cudaMalloc(&d_puncturedVecRand.data, d_randMatrix.rows / 8);

    dim3 grid(8, 256);
    dim3 block(512);
    for (int chunkR = 0; chunkR < d_randMatrix.rows / CHUNK_SIDE; chunkR++) {
        for (int chunkC = 0; chunkC < d_randMatrix.cols / CHUNK_SIDE; chunkC++) {
            mat_vec_mult<<<grid, block>>>(d_choiceVecRand, d_subTotalChoice,
                d_randMatrix, d_choiceVec, chunkC * CHUNK_SIDE);
            mat_vec_mult<<<grid, block>>>(d_puncturedVecRand, d_subTotalPunctured,
                d_randMatrix, d_puncturedVec, chunkC * CHUNK_SIDE);
            cudaDeviceSynchronize();
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    float duration = (end.tv_sec - start.tv_sec) * 1000;
    duration += (end.tv_nsec - start.tv_nsec) / 1000000.0;
    printf("Matrix mult recver using GPU: %0.4f ms\n", duration / NUM_SAMPLES);
}
