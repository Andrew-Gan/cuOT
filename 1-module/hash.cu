#include <algorithm>
#include "hash.h"

/************************************************************
Algorithm generate chunks of full matrix and pass into kernel
An example:

Tier    | Dimension (bits)  | Size
Matrix  | 2^20 x 2^21       | 256 GB
Chunk   | 2^18 x 2^18       |   8 GB
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
void mat_vec_hash(uint8_t *out, uint8_t *subTotal, Matrix matrix, uint8_t *in, int numRows, int globalCol) {
  int chunkStartRow = blockIdx.y * numRows;
  int col_byte = blockIdx.x * blockDim.x + threadIdx.x;

  for (int row = chunkStartRow; row < chunkStartRow + numRows; row++) {
    if (in[row / 8] & (1 << (row % 8)) != 0) {
      subTotal[blockIdx.y * (matrix.cols / 8) + col_byte]
       ^= matrix.data[row * (matrix.cols / 8) + col_byte];
    }
  }
  if (blockIdx.y == 0) {
    for(int i = 0; i < gridDim.y; i++) {
      out[globalCol/8+col_byte] ^= subTotal[i*matrix.cols/8+col_byte];
    }
  }
}

__global__
void mat_sparse_vec_hash(uint8_t *out, Matrix matrix, SparseVector vec, int globalRow, int globalCol) {
  int col_byte = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t t = 0; t < vec.weight; t++) {
    size_t globalRow = vec.nonZeros[t];
    if (globalRow > globalRow && globalRow < globalRow + matrix.rows) {
      size_t localRow = globalRow - globalRow;
      out[matrix.cols/8+col_byte] ^= matrix.data[localRow*matrix.cols/8+col_byte];
    }
  }
}

__host__
void hash_sender(GPUBlock &fullVectorHashed, Matrix &randMatrix,
  GPUBlock &fullVector, int chunkC) {

  EventLog::start(HashSender);
  size_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  size_t numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile / 8);
  GPUBlock subTotal(grid.y * randMatrix.cols / 8);

  mat_vec_hash<<<grid, block>>>(fullVectorHashed.data_d, subTotal.data_d, randMatrix, fullVector.data_d, numRowsPerTile, chunkC * randMatrix.cols);
  cudaDeviceSynchronize();
  EventLog::end(HashSender);
}

__host__
void hash_recver(GPUBlock &puncVectorHashed, GPUBlock &choiceVectorHashed,
  Matrix &randMatrix, GPUBlock &puncVector, SparseVector &choiceVector,
  int chunkR, int chunkC) {

  EventLog::start(HashRecver);
  size_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  int numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile / 8);
  GPUBlock subTotal(grid.y * randMatrix.cols / 8);

  size_t globalRow = chunkR * randMatrix.rows;
  size_t globalCol = chunkC * randMatrix.cols;
  dim3 gridSparse(randMatrix.cols / 1024);

  mat_vec_hash<<<grid, block>>>(puncVectorHashed.data_d, subTotal.data_d, randMatrix, puncVector.data_d, numRowsPerTile, globalCol);
  mat_sparse_vec_hash<<<gridSparse, 1024>>>(choiceVectorHashed.data_d, randMatrix, choiceVector, globalRow, globalCol);
  cudaDeviceSynchronize();
  EventLog::end(HashRecver);
}
