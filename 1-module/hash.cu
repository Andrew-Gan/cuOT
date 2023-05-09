#include "hash.h"
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
void mat_vec_hash(Vector out, uint8_t *subTotal, Matrix matrix, Vector vec, int numRows, int globalStartCol) {
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
void hash_sender(Matrix randMatrix_d, Vector fullVec_d, int chunkC) {
  // for when matrix size < tile
  size_t numRowsPerTile = std::min(randMatrix_d.rows, TILE_H);
  int numColsPerTile = std::min(randMatrix_d.cols / 8, TILE_W / 8);
  dim3 grid((randMatrix_d.cols-1) / TILE_W + 1, (randMatrix_d.rows-1) / TILE_H + 1);
  dim3 block(numColsPerTile);

  uint8_t *subTotal_d;
  cudaMalloc(&subTotal_d, grid.y * randMatrix_d.cols / 8);
  Vector randomVec_d = { .n = randMatrix_d.cols };
  cudaMalloc(&randomVec_d.data, randomVec_d.n / 8);

  mat_vec_hash<<<grid, block>>>(randomVec_d, subTotal_d, randMatrix_d,
    fullVec_d, numRowsPerTile, chunkC * randMatrix_d.cols);
  cudaDeviceSynchronize();
}

__host__
void hash_recver(Matrix randMatrix_d, Vector choiceVec_d, Vector puncVec_d, int chunkC) {
  // for when matrix size < tile
  size_t numRowsPerTile = std::min(randMatrix_d.rows, TILE_H);
  int numColsPerTile = std::min(randMatrix_d.cols / 8, TILE_W / 8);
  dim3 grid((randMatrix_d.cols-1) / TILE_W + 1, (randMatrix_d.rows-1) / TILE_H + 1);
  dim3 block(numColsPerTile);

  uint8_t *subTotalChoice_d, *subTotalPunctured_d;
  cudaMalloc(&subTotalChoice_d, grid.y * randMatrix_d.cols / 8);
  cudaMalloc(&subTotalPunctured_d, grid.y * randMatrix_d.cols / 8);
  Vector choiceVecRand_d = { .n = randMatrix_d.cols };
  Vector puncVecRand_d  =  { .n = randMatrix_d.cols };
  cudaMalloc(&choiceVecRand_d.data, choiceVecRand_d.n / 8);
  cudaMalloc(&puncVecRand_d.data, puncVecRand_d.n / 8);

  mat_vec_hash<<<grid, block>>>(choiceVecRand_d, subTotalChoice_d,
    randMatrix_d, choiceVec_d, numRowsPerTile, chunkC * randMatrix_d.cols);
  mat_vec_hash<<<grid, block>>>(puncVecRand_d, subTotalPunctured_d,
    randMatrix_d, puncVec_d, numRowsPerTile, chunkC * randMatrix_d.cols);
  cudaDeviceSynchronize();
}
