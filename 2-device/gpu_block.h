#ifndef __GPU_BLOCK_H__
#define __GPU_BLOCK_H__

#include "util.h"

class GPUBlock {
public:
  GPUBlock();
  GPUBlock(size_t n);
  GPUBlock(const GPUBlock &blk);
  virtual ~GPUBlock();
  uint8_t *data_d = nullptr;
  size_t nBytes = 0;
  GPUBlock operator*(uint8_t scalar);
  GPUBlock operator^(const GPUBlock &rhs);
  GPUBlock& operator=(const GPUBlock &rhs);
  bool operator==(const GPUBlock &rhs);
  bool operator!=(const GPUBlock &rhs);
  uint8_t& operator[](int index);
  void set(uint32_t val);
  void set(const uint8_t *val, size_t n);
};

std::ostream& operator<<(std::ostream &os, const GPUBlock &obj);

#endif
