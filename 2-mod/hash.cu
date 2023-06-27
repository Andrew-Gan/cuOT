#include <algorithm>
#include "silentOT.h"

/************************************************************
Algorithm generate chunks of full matrix and pass into kernel
For 1 million requested OTs:

Tier    | Dimension (bits)  | Size
Matrix  | 2^20 x 2^21       | 256 GB
Chunk   | 2^18 x 2^18       |   8 GB
Tile    | 2^12  x 2^10      | 512 KB

1 full matrix   = 4x8 chunks
1 chunk         = 256x256 tiles / blocks
1 tile / block  = 32 warps
************************************************************/

#define TILE_H (uint64_t) 4096
#define TILE_W (uint64_t) 1024 // at most 1024

__global__
void mat_vec_hash(uint8_t *out, uint8_t *subTotal, GPUMatrix<OTBlock> matrix, uint8_t *in, int numRows, int globalCol) {
  // int chunkStartRow = blockIdx.y * numRows;
  // int col_byte = blockIdx.x * blockDim.x + threadIdx.x;

  // for (int row = chunkStartRow; row < chunkStartRow + numRows; row++) {
  //   if (in[row / 8] & (1 << (row % 8)) != 0) {
  //     subTotal[blockIdx.y * (matrix.cols / 8) + col_byte]
  //      ^= matrix.data[row * (matrix.cols / 8) + col_byte];
  //   }
  // }
  // if (blockIdx.y == 0) {
  //   for(int i = 0; i < gridDim.y; i++) {
  //     out[globalCol/8+col_byte] ^= subTotal[i*matrix.cols/8+col_byte];
  //   }
  // }
}

__global__
void mat_sparse_vec_hash(uint8_t *out, GPUMatrix<OTBlock> matrix, SparseVector vec, int globalRow, int globalCol) {
  // int col_byte = blockIdx.x * blockDim.x + threadIdx.x;
  // for (uint64_t t = 0; t < vec.weight; t++) {
  //   uint64_t globalRow = vec.nonZeros[t];
  //   if (globalRow > globalRow && globalRow < globalRow + matrix.rows) {
  //     uint64_t localRow = globalRow - globalRow;
  //     out[matrix.cols/8+col_byte] ^= matrix.data[localRow*matrix.cols/8+col_byte];
  //   }
  // }
}

__host__
void SilentOTSender::compress(GPUBlock &fullVectorHashed, GPUMatrix<OTBlock> &randMatrix,
  GPUBlock &fullVector, int chunkC) {

  EventLog::start(Sender, Hash);
  uint64_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  uint64_t numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile);
  GPUBlock subTotal(grid.y * randMatrix.cols);

  mat_vec_hash<<<grid, block>>>(fullVectorHashed.data_d, subTotal.data_d, randMatrix, fullVector.data_d, numRowsPerTile, chunkC * randMatrix.cols);
  cudaDeviceSynchronize();
  EventLog::end(Sender, Hash);
}

__host__
void SilentOTRecver::compress(GPUBlock &puncVectorHashed, GPUBlock &choiceVectorHashed,
  GPUMatrix<OTBlock> &randMatrix, GPUBlock &puncVector, SparseVector &choiceVector,
  int chunkR, int chunkC) {

  EventLog::start(Recver, Hash);
  uint64_t numRowsPerTile = std::min(randMatrix.rows, TILE_H);
  int numColsPerTile = std::min(randMatrix.cols, TILE_W);
  dim3 grid(randMatrix.cols / numColsPerTile, randMatrix.rows / numRowsPerTile);
  dim3 block(numColsPerTile / 8);
  GPUBlock subTotal(grid.y * randMatrix.cols / 8);

  uint64_t globalRow = chunkR * randMatrix.rows;
  uint64_t globalCol = chunkC * randMatrix.cols;
  dim3 gridSparse(randMatrix.cols / 1024);

  mat_vec_hash<<<grid, block>>>(puncVectorHashed.data_d, subTotal.data_d, randMatrix, puncVector.data_d, numRowsPerTile, globalCol);
  mat_sparse_vec_hash<<<gridSparse, 1024>>>(choiceVectorHashed.data_d, randMatrix, choiceVector, globalRow, globalCol);
  cudaDeviceSynchronize();
  EventLog::end(Recver, Hash);
}
