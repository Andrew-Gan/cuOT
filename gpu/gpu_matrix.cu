#include <iomanip>
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
      sprintf(msg, "Mat::data: exceed dim at %d, accessing %lu when max is %lu\n", i, pos.at(i), mDim.at(i));
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
  cudaMemcpyAsync(tpBuffer, mPtr, mNBytes, cudaMemcpyDeviceToDevice);
  dim3 block, grid;
  uint64_t threadX = col * sizeof(blk);
  block.x = std::min(threadX, 32UL);
  grid.x = (threadX + block.x - 1) / block.x;
  uint64_t threadY = row / 8;
  block.y = std::min(threadY, 32UL);
  uint64_t yBlock = (threadY + block.y - 1) / block.y;
  grid.y = std::min(yBlock, 32768UL);
  grid.z = (yBlock + grid.y - 1) / grid.y;
  bit_transposer<<<grid, block>>>(mPtr, tpBuffer);
  cudaFreeAsync(tpBuffer, 0);
  uint64_t tpRows = col * 8 * sizeof(blk);
  mDim.at(1) = row / (8 * sizeof(blk));
  mDim.at(0) = tpRows;
}

void Mat::modp(uint64_t reducedCol) {
  if (mDim.size() > 2)
    throw std::invalid_argument("Mat::modp only 1D or 2D matrix supported\n");

  uint64_t col = mDim.back();
  uint64_t threads = reducedCol * sizeof(blk);
  uint64_t block = std::min(threads, 1024lu);
  uint64_t rows = mDim.size() == 2 ? mDim.front() : 1;
  dim3 grid = dim3((threads + block - 1) / block, rows);

  for (uint64_t i = 1; i < col / reducedCol; i++)
    gpu_xor<<<grid, block>>>(mPtr, mPtr + (i*threads), threads, col*sizeof(blk));
}

void Mat::xor_scalar(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  xor_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
}

Mat& Mat::operator&=(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  and_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
  return *this;
}

std::ostream& operator<<(std::ostream &os, Mat &obj) {
  if (obj.dims().size() > 2)
    throw std::invalid_argument("Mat::operator<< only 1D or 2D matrix supported\n");
  blk *tmp = new blk[obj.size_bytes() / sizeof(blk)];
  uint64_t rows = obj.dims().size() == 2 ? obj.dim(0) : 1;
  uint64_t cols = obj.dims().size() == 2 ? obj.dim(1) : obj.dim(0);
  cudaMemcpy(tmp, obj.data(), obj.size_bytes(), cudaMemcpyDeviceToHost);
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      blk *val = tmp+i*cols+j;
      for (int i = 0; i < 1; i++) {
        os << std::setw(8) << std::setfill('0') << std::hex << val->data[i];
      }
      os << " ";
    }
    os << std::endl;
  }
  os << std::dec;
  delete[] tmp;
  return os;
}
