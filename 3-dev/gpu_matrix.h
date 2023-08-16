#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include <bitset>
#include "gpu_data.h"
#include "gpu_ops.h"

template<typename T>
class GPUmatrix : public GPUdata {
public:
  GPUmatrix(uint64_t r, uint64_t c);
  uint64_t rows() const { return mRows; }
  uint64_t cols() const { return mCols; }
  T* data() const { return (T*) mPtr; }
  void set(uint64_t r, uint64_t c, T &val);
  void resize(uint64_t r, uint64_t c);
  void bit_transpose();
  void modp(uint64_t reducedTerms);
  void xor_one_to_many_async(T *rhs, cudaStream_t s);
  GPUmatrix<T>& operator&=(T *rhs);

protected:
  uint64_t mRows, mCols;
};

template<typename T>
GPUmatrix<T>::GPUmatrix(uint64_t r, uint64_t c) : GPUdata(r * c * sizeof(T)) {
  mRows = r;
  mCols = c;
}

template<typename T>
void GPUmatrix<T>::set(uint64_t r, uint64_t c, T &val) {
  uint64_t offset = r * mCols + c;
  cudaMemcpy((T*) mPtr + offset, &val, sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPUmatrix<T>::resize(uint64_t r, uint64_t c) {
  GPUdata::resize(r*c*sizeof(T));
  mRows = r;
  mCols = c;
}

template<typename T>
void GPUmatrix<T>::bit_transpose() {
  uint8_t *tpBuffer;
  cudaMalloc(&tpBuffer, mNBytes);
  dim3 block, grid;
  if (mCols * sizeof(T) < 32) {
    block.x = mCols * sizeof(T);
    grid.x = 1;
  }
  else {
    block.x = 32;
    grid.x = mCols * sizeof(T) / 32;
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
  uint64_t tpRows = mCols * 8 * sizeof(T);
  mCols = mRows / (8 * sizeof(T));
  mRows = tpRows;
}

template<typename T>
void GPUmatrix<T>::modp(uint64_t reducedTerms) {
  uint64_t block = std::min(reducedTerms, 1024lu);
  uint64_t grid = reducedTerms < 1024 ? 1 : (reducedTerms + 1023) / 1024;
  poly_mod_gpu<<<grid, block>>>((uint64_t*) mPtr, mCols);
  cudaDeviceSynchronize();
  mCols = reducedTerms;
}

template<typename T>
void GPUmatrix<T>::xor_one_to_many_async(T *rhs, cudaStream_t s) {
  uint64_t nBlk = (mNBytes + 1023) / 1024;
  xor_single_gpu<<<nBlk, 1024, 0, s>>>(mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
}

template<typename T>
GPUmatrix<T>& GPUmatrix<T>::operator&=(T *rhs) {
  uint64_t nBlk = (mNBytes + 1023) / 1024;
  and_single_gpu<<<nBlk, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
  cudaDeviceSynchronize();
}

#endif
