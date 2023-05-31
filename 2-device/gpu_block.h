#ifndef __GPU_BLOCK_H__
#define __GPU_BLOCK_H__

#include "util.h"

class GPUBlock {
public:
  GPUBlock();
  GPUBlock(size_t n);
  GPUBlock(const GPUBlock &blk);
  GPUBlock(const SparseVector &vec, size_t stretch);
  virtual ~GPUBlock();
  uint8_t *data_d = nullptr;
  size_t nBytes = 0;
  GPUBlock operator*(const GPUBlock &rhs);
  GPUBlock operator^(const GPUBlock &rhs);
  GPUBlock& operator=(const GPUBlock &rhs);
  GPUBlock& operator^=(const GPUBlock &rhs);
  bool operator==(const GPUBlock &rhs);
  bool operator!=(const GPUBlock &rhs);
  uint8_t& operator[](int index);
  void set(uint32_t val);
  void set(const uint8_t *val, size_t n);
  GPUBlock sum(size_t elemSize);
  void resize(size_t size);
  void append(GPUBlock &rhs);
};

std::ostream& operator<<(std::ostream &os, const GPUBlock &obj);

#endif
