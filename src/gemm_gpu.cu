#include "gemm_gpu.h"
#include <algorithm>

/************************************************************
Algorithm generate chunks of full matrix and pass into kernel
An example:

Tier    | Dimension (bits)  | Size
Matrix  | 2^20 x 2^21       | 256 GB
Chunk   | 2^17 x 2^17       |   2 GB
Tile    | 2^7  x 2^12       |  64 KB
*no shared mem needed for tile
*chunk size defined in util.h

1 full matrix   = 32x64 chunks
1 chunk         = 32x64 tblocks
1 tile / block  = 16 warps
16 warps        = 512 threads
************************************************************/

#define TILE_H (size_t)128
#define TILE_W (size_t)4096
#define T_PER_BLK (size_t)512

__global__
void mat_vec_mult(Vector out, uint8_t *subTotal, Matrix matrix, Vector vec, int numRows, int globalStartCol) {
  // treat the unirand mat as transposed
  // uniform rand mat ~ transposed
  // accessing by row in transposed = accessing by col in original
  // threads in same warp access same row for coalescing
  int startRow = blockIdx.y * numRows;
  int col_byte = blockIdx.x * blockDim.x + threadIdx.x;

  for (int row = startRow; row < startRow + numRows; row++) {
    if (vec.data[row / 8] & (1 << (row % 8)) != 0) {
      subTotal[blockIdx.y * (matrix.cols / 8) + col_byte]
       ^= matrix.data[row * (matrix.cols / 8) + col_byte];
    }
  }
  if (blockIdx.y == 0) {
    for(int i = 0; i < gridDim.y; i++) {
      out.data[globalStartCol+col_byte] ^= subTotal[i*matrix.cols/8+col_byte];
    }
  }
}

__host__
void mult_sender_gpu(Matrix d_randMatrix, Vector d_fullVec, int chunkC) {
  // for when matrix size < tile
  size_t numRowsPerTile = std::min(d_randMatrix.rows, TILE_H);
  int numColsPerTile = std::min(d_randMatrix.cols / 8, TILE_W / 8);
  dim3 grid((d_randMatrix.cols-1) / TILE_W + 1, (d_randMatrix.rows-1) / TILE_H + 1);
  dim3 block(numColsPerTile);

  uint8_t *d_subTotal;
  cudaMalloc(&d_subTotal, grid.y * d_randMatrix.cols / 8);
  Vector d_randomVec = { .n = d_randMatrix.cols };
  cudaMalloc(&d_randomVec.data, d_randomVec.n / 8);

  mat_vec_mult<<<grid, block>>>(d_randomVec, d_subTotal, d_randMatrix,
    d_fullVec, numRowsPerTile, chunkC * d_randMatrix.cols);
  cudaDeviceSynchronize();
}

__host__
void mult_recver_gpu(Matrix d_randMatrix, Vector d_choiceVec, Vector d_puncVec, int chunkC) {
  // for when matrix size < tile
  size_t numRowsPerTile = std::min(d_randMatrix.rows, TILE_H);
  int numColsPerTile = std::min(d_randMatrix.cols / 8, TILE_W / 8);
  dim3 grid((d_randMatrix.cols-1) / TILE_W + 1, (d_randMatrix.rows-1) / TILE_H + 1);
  dim3 block(numColsPerTile);

  uint8_t *d_subTotalChoice, *d_subTotalPunctured;
  cudaMalloc(&d_subTotalChoice, grid.y * d_randMatrix.cols / 8);
  cudaMalloc(&d_subTotalPunctured, grid.y * d_randMatrix.cols / 8);
  Vector d_choiceVecRand = { .n = d_randMatrix.cols };
  Vector d_puncVecRand  =  { .n = d_randMatrix.cols };
  cudaMalloc(&d_choiceVecRand.data, d_choiceVecRand.n / 8);
  cudaMalloc(&d_puncVecRand.data, d_puncVecRand.n / 8);

  mat_vec_mult<<<grid, block>>>(d_choiceVecRand, d_subTotalChoice,
    d_randMatrix, d_choiceVec, numRowsPerTile, chunkC * d_randMatrix.cols);
  mat_vec_mult<<<grid, block>>>(d_puncVecRand, d_subTotalPunctured,
    d_randMatrix, d_puncVec, numRowsPerTile, chunkC * d_randMatrix.cols);
  cudaDeviceSynchronize();
}
