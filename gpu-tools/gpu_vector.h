#ifndef __GPU_VECTOR_H__
#define __GPU_VECTOR_H__

#include "gpu_matrix.h"

class GPUvector : public GPUmatrix {
public:
  using GPUdata::xor_d;
  using GPUmatrix::data;

  GPUvector() {}
  GPUvector(uint64_t len) : GPUmatrix(1, len) {}
  uint64_t size() { return mCols; }
  blk* data(uint64_t i) const;
  void set(uint64_t i, blk &val) { GPUmatrix::set(0, i, val); }
  void resize(uint64_t len) { GPUmatrix::resize(1, len); }
  void sum(uint64_t nPartition, uint64_t blkPerPart);
  void xor_d(GPUvector &rhs, uint64_t offs);
};

using vec = GPUvector;

#endif
