#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include "gpu_tools.h"
#include "gpu_data.h"

class GPUmatrix : public GPUdata {
public:
  GPUmatrix() {}
  GPUmatrix(uint64_t r, uint64_t c);
  uint64_t rows() const { return mRows; }
  uint64_t cols() const { return mCols; }
  blk* data() const { return (blk*) mPtr; }
  blk* data(uint64_t r, uint64_t c) const { return (blk*)mPtr + (r * mRows + c); }
  void set(uint64_t r, uint64_t c, blk &val);
  void resize(uint64_t r, uint64_t c);
  void bit_transpose();
  void modp(uint64_t reducedCol);
  void xor_scalar(blk *rhs);
  GPUmatrix& operator&=(blk *rhs);
  GPUmatrix& operator%=(uint64_t mod);
  void print_bits(const char *filename);

protected:
  uint64_t mRows = 0, mCols = 0;
};

using mat = GPUmatrix;

#endif
