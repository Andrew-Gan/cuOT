#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include "gpu_define.h"
#include "gpu_data.h"

#include <vector>

class Mat : public GPUdata {
public:
  Mat() {}
  Mat(const Mat &other);
  Mat(std::vector<uint64_t> newDim);
  std::vector<uint64_t> dims() const { return mDim; }
  uint64_t dim(uint32_t i) const;
  blk* data() const { return (blk*) mPtr; }
  blk* data(std::vector<uint64_t> pos) const;
  void set(blk &val, std::vector<uint64_t> pos);
  void resize(std::vector<uint64_t> newDim);
  void xor_scalar(blk *rhs);
  Mat& operator&=(blk *rhs);
  Mat& operator%=(uint64_t mod);
  uint64_t size() const { return listToSize(mDim); }

  // 2D Matrix only
  void bit_transpose();
  void modp(uint64_t reducedCol);

private:
  std::vector<uint64_t> mDim;
  static uint64_t listToSize(std::vector<uint64_t> dim);
  uint64_t listToOffset(std::vector<uint64_t> pos) const;
};

#endif
