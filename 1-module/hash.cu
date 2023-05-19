#include <algorithm>
#include "hash.h"

/************************************************************
Algorithm generate chunks of full matrix and pass into kernel
An example:

Tier    | Dimension (bits)  | Size
Matrix  | 2^20 x 2^21       | 256 GB
Chunk   | 2^17 x 2^17       |   2 GB
Tile    | 2^9  x 2^10       |  64 KB
*no shared mem needed for tile
*chunk size defined in util.h

1 full matrix   = 32x64 chunks
1 chunk         = 32x64 tblocks
1 tile / block  = 16 warps
16 warps        = 512 threads
************************************************************/

#define TILE_H (size_t) 512
#define TILE_W (size_t) 1024
#define T_PER_BLK (size_t) 512

__global__
void mat_vec_hash(Vector out, uint8_t *subTotal, Matrix matrix, Vector vec, int numRows, int globalStartCol) {
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
void hash_sender(Matrix randMatrix, Vector fullVec, int chunkC) {
  EventLog::start(HashSender);
  size_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  size_t numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile / 8);
  uint8_t *subTotal_d;
  cudaMalloc(&subTotal_d, grid.y * randMatrix.cols / 8);
  Vector randomVec_d = { .n = randMatrix.cols };
  cudaMalloc(&randomVec_d.data, randomVec_d.n / 8);

  mat_vec_hash<<<grid, block>>>(randomVec_d, subTotal_d, randMatrix,
    fullVec, numRowsPerTile, chunkC * randMatrix.cols);
  cudaDeviceSynchronize();
  EventLog::end(HashSender);

  cudaFree(subTotal_d);
  cudaFree(randomVec_d.data);
}

__host__
void hash_recver(Matrix randMatrix, Vector choiceVec, Vector puncVec, int chunkC) {
  EventLog::start(HashRecver);
  size_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  int numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile / 8);
  uint8_t *subTotalChoice_d, *subTotalPunctured_d;
  cudaMalloc(&subTotalChoice_d, grid.y * randMatrix.cols / 8);
  cudaMalloc(&subTotalPunctured_d, grid.y * randMatrix.cols / 8);
  Vector choiceVecRand_d = { .n = randMatrix.cols };
  Vector puncVecRand_d  =  { .n = randMatrix.cols };
  cudaMalloc(&choiceVecRand_d.data, choiceVecRand_d.n / 8);
  cudaMalloc(&puncVecRand_d.data, puncVecRand_d.n / 8);

  mat_vec_hash<<<grid, block>>>(choiceVecRand_d, subTotalChoice_d,
    randMatrix, choiceVec, numRowsPerTile, chunkC * randMatrix.cols);
  mat_vec_hash<<<grid, block>>>(puncVecRand_d, subTotalPunctured_d,
    randMatrix, puncVec, numRowsPerTile, chunkC * randMatrix.cols);
  cudaDeviceSynchronize();
  EventLog::end(HashRecver);

  cudaFree(subTotalChoice_d);
  cudaFree(subTotalPunctured_d);
  cudaFree(choiceVecRand_d.data);
  cudaFree(puncVecRand_d.data);
}
