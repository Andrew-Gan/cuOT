#ifndef __GPU_BLOCK_H__
#define __GPU_BLOCK_H__

#include "util.h"

class GPUBlock {
public:
  GPUBlock();
  GPUBlock(uint64_t n);
  GPUBlock(const GPUBlock &blk);
  GPUBlock(const SparseVector &vec, uint64_t stretch);
  virtual ~GPUBlock();
  uint8_t *data_d = nullptr;
  uint64_t nBytes = 0;
  GPUBlock& operator=(const GPUBlock &rhs);
  GPUBlock& operator*=(const GPUBlock &rhs);
  GPUBlock& operator^=(const GPUBlock &rhs);
  bool operator==(const GPUBlock &rhs);
  bool operator!=(const GPUBlock &rhs);
  uint8_t& operator[](int index);
  void clear();
  void set(uint64_t val);
  void set(const uint8_t *val, uint64_t n);
  void set(const uint8_t *val, uint64_t n, uint64_t offset);
  void sum_async(uint64_t elemSize);
  void resize(uint64_t size);
  void append(GPUBlock &rhs);
  void minCopy(GPUBlock &rhs);
};

std::ostream& operator<<(std::ostream &os, const GPUBlock &obj);

#endif
