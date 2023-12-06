#include "gpu_ops.h"
#include "gpu_vector.h"
#include <cstdio>
#include <stdexcept>

blk* Vec::data(uint64_t i) const {
  if (i >= dim(1)) {
    char msg[40];
    sprintf(msg, "Accessing index %lu in Vec of len %lu\n", i, dim(1));
    throw std::invalid_argument(msg);
  }
  return Mat::data({0, i});
}

// nPartition: number of partitions to reduce to separate totals
// blkPerPart: number of blks in a partition
void Vec::sum(uint64_t nPartition, uint64_t blkPerPart) {
  uint64_t blockSize, nBlocks, mem;

  uint8_t *buffer;
  cudaMalloc(&buffer, mNBytes);

  uint64_t *in = (uint64_t*) buffer;
  uint64_t *out = (uint64_t*) this->mPtr;

  for (uint64_t remBlocks = blkPerPart; remBlocks > 1; remBlocks /= 1024) {
    std::swap(in, out);

    blockSize = std::min((uint64_t) 1024, remBlocks);
    nBlocks = nPartition * (remBlocks / blockSize);
    mem = blockSize * sizeof(uint64_t);
    xor_reduce<<<nBlocks, blockSize, mem>>>(out, in);
  }

  if (out != (uint64_t*) this->mPtr) {
    cudaMemcpy(this->mPtr, out, nPartition * sizeof(blk), cudaMemcpyDeviceToDevice);
  }

  cudaFree(buffer);
}

void Vec::xor_d(Vec &rhs, uint64_t offs) {
  uint64_t min = std::min(this->mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(this->mPtr, (uint8_t*) (rhs.data(offs)), min);
}

Span Vec::span(uint64_t start, uint64_t end) {
  return Span(*this, start, end == 0 ? size() : end);
}

Span::Span(Vec &data, uint64_t start, uint64_t end) : obj(data) {
  if (start >= data.size() || end > data.size()) {
    throw std::invalid_argument("Span::Span() Range exceeds Vec dim\n");
  }

  this->start = start;
  this->range = end - start;
}

blk* Span::data(uint64_t i) const {
  if (i >= range)
    throw std::invalid_argument("Span::data() Index exceeds Span dim\n");

  return obj.data(start + i);
};

void Span::set(uint64_t i, blk &val) {
  if (i >= range)
    throw std::invalid_argument("Span::set() Index exceeds Span dim\n");

  obj.set(start + i, val);
}

void Span::operator=(const Span& other) {
  if (size() != other.size())
    throw std::invalid_argument("Span::operator=() Unequal span len is unsupported\n");

  cudaMemcpy(data(), other.data(), size()*sizeof(blk), cudaMemcpyDeviceToDevice);
}
