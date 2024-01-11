#include <bitset>
#include "gpu_ops.h"
#include "gpu_matrix.h"
#include <stdexcept>
#include "gpu_tests.h"

Mat::Mat(std::vector<uint64_t> newDim) : GPUdata(listToSize(newDim)*sizeof(blk)) {
  mDim = newDim;
}

Mat::Mat(const Mat &other) : GPUdata(other) {
  mDim = other.mDim;
}

uint64_t Mat::dim(uint32_t i) const {
  if (mDim.size() == 0)
    return 0;
  if (i >= mDim.size()) {
    throw std::invalid_argument("Requested dim exceeds matrix dim\n");
  }
  return mDim.at(i);
}

uint64_t Mat::listToSize(std::vector<uint64_t> dim) {
  uint64_t size = 1;
  for (const uint64_t &i : dim) {
    size *= i;
  }
  return size;
}

uint64_t Mat::listToOffset(std::vector<uint64_t> pos) const {
  if (pos.size() != mDim.size())
    throw std::invalid_argument("Matrix dim and pos len mismatch\n");

  uint64_t offs = 0;
  for (int i = 0; i < pos.size() - 1; i++) {
    offs += pos.at(i) * mDim.at(i+1);
  }
  offs += pos.back();
  return offs;
}

blk* Mat::data(std::vector<uint64_t> pos) const {
  if (pos.size() != mDim.size())
    throw std::invalid_argument("Matrix dim and pos dim mismatch\n");

  for (int i = 0; i < pos.size(); i++) {
    if (pos.at(i) >= mDim.at(i)) {
      char msg[40];
      sprintf(msg, "Requested dim exceed matrix dim at %d\n", i);
      throw std::invalid_argument(msg);
    }
  }
  
  return (blk*)mPtr + listToOffset(pos);
}

void Mat::set(blk &val, std::vector<uint64_t> pos) {
  uint64_t offset = listToOffset(pos);
  cudaMemcpy((blk*)mPtr + offset, &val, sizeof(blk), cudaMemcpyHostToDevice);
}

void Mat::resize(std::vector<uint64_t> newDim) {
  GPUdata::resize(listToSize(newDim)*sizeof(blk));
  mDim = newDim;
}

void Mat::bit_transpose() {
  if (mDim.size() != 2)
    throw std::invalid_argument("Mat::bit_transpose only 2D matrix supported\n");

  uint64_t row = dim(0);
  uint64_t col = dim(1);
  if (row < 8 * sizeof(blk)) 
    throw std::invalid_argument("Mat::bit_transpose insufficient rows to transpose\n");

  uint8_t *tpBuffer;
  cudaMalloc(&tpBuffer, mNBytes);
  dim3 block, grid;
  uint64_t threadX = col * sizeof(blk);
  block.x = std::min(threadX, 32UL);
  grid.x = (threadX + block.x - 1) / block.x;
  uint64_t threadY = row / 8;
  block.y = std::min(threadY, 32UL);
  uint64_t yBlock = (threadY + block.y - 1) / block.y;
  grid.y = std::min(yBlock, 32768UL);
  grid.z = (yBlock + grid.y - 1) / grid.y;

  bit_transposer<<<grid, block>>>(tpBuffer, mPtr, grid);
  check_call("Mat::bit_transpose\n");
  cudaFree(mPtr);
  mPtr = tpBuffer;
  uint64_t tpRows = col * 8 * sizeof(blk);
  col = row / (8 * sizeof(blk));
  row = tpRows;
}

void Mat::modp(uint64_t reducedCol) {
  if (mDim.size() != 2)
    throw std::invalid_argument("Mat::modp only 2D matrix supported\n");

  uint64_t row = dim(0);
  uint64_t col = dim(1);
  uint64_t block = std::min(reducedCol, 1024lu);
  uint64_t grid = (reducedCol + block - 1) / block;

  for (uint64_t i = 0; i < col / reducedCol - 1; i++) {
    gpu_xor<<<grid, block>>>(mPtr, mPtr + (i * reducedCol * sizeof(blk)), reducedCol);
  }
  check_call("Mat::modp\n");

  col = reducedCol;
}

void Mat::xor_scalar(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  xor_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
  check_call("Mat::xor_scalar\n");
}

Mat& Mat::operator&=(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  and_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
  check_call("Mat::operator&=\n");

  return *this;
}
