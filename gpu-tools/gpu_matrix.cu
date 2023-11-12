#include <bitset>
#include "gpu_ops.h"
#include "gpu_matrix.h"

GPUmatrix::GPUmatrix(uint64_t r, uint64_t c) : GPUdata(r * c * sizeof(blk)) {
  mRows = r;
  mCols = c;
}

void GPUmatrix::set(uint64_t r, uint64_t c, blk &val) {
  uint64_t offset = r * mCols + c;
  cudaMemcpy((blk*) mPtr + offset, &val, sizeof(blk), cudaMemcpyHostToDevice);
}

void GPUmatrix::resize(uint64_t r, uint64_t c) {
  GPUdata::resize(r*c*sizeof(blk));
  mRows = r;
  mCols = c;
}

void GPUmatrix::bit_transpose() {
  if (mRows < 8 * sizeof(blk)) {
    printf("GPUmatrix::bit_transpose() insufficient rows to transpose\n");
    return;
  }
  uint8_t *tpBuffer;
  cudaMalloc(&tpBuffer, mNBytes);
  dim3 block, grid;
  if (mCols * sizeof(blk) < 32) {
    block.x = mCols * sizeof(blk);
    grid.x = 1;
  }
  else {
    block.x = 32;
    grid.x = mCols * sizeof(blk) / 32;
  }
  if (mRows / 8 < 32) {
    block.y = mRows / 8;
    grid.y = 1;
  }
  else {
    block.y = 32;
    grid.y = mRows / 8 / 32;
  }
  // translate 2D grid into 1D due to CUDA limitations
  bit_transposer<<<grid.x * grid.y, block>>>(tpBuffer, mPtr, grid);
  cudaDeviceSynchronize();
  cudaFree(mPtr);
  mPtr = tpBuffer;
  uint64_t tpRows = mCols * 8 * sizeof(blk);
  mCols = mRows / (8 * sizeof(blk));
  mRows = tpRows;
}

void GPUmatrix::modp(uint64_t reducedCol) {
  uint64_t block = std::min(reducedCol, 1024lu);
  uint64_t grid = reducedCol < 1024 ? 1 : (reducedCol + 1023) / 1024;
  for (uint64_t i = 0; i < mCols / reducedCol; i++) {
    gpu_xor<<<grid, block>>>(mPtr, mPtr + (i * reducedCol * sizeof(blk)), mCols);
    cudaDeviceSynchronize();
  }
  mCols = reducedCol;
}

void GPUmatrix::xor_scalar(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  xor_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
}

GPUmatrix& GPUmatrix::operator&=(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  and_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
  cudaDeviceSynchronize();
}

void GPUmatrix::print_bits(const char *filename) {
  std::ofstream ofs(filename);
  uint8_t *cpuBuffer = new uint8_t[mNBytes];
  cudaMemcpy(cpuBuffer, mPtr, mNBytes, cudaMemcpyDeviceToHost);
  for (uint64_t r = 0; r < mRows; r++) {
    for (uint64_t c = 0; c < mCols * sizeof(blk); c++) {
      uint64_t offset = r * mCols * sizeof(blk) + c;
      std::bitset<8> byteBits(cpuBuffer[offset]);
      ofs << byteBits << " ";
    }
    ofs << std::endl;
  }
  delete[] cpuBuffer;
}
