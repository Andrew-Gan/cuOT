#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include "gpu_data.h"
#include "gpu_ops.h"

template<typename T>
class GPUmatrix : public GPUdata {
public:
  using GPUdata::xor_async;
  GPUmatrix() {}
  GPUmatrix(uint64_t r, uint64_t c);
  uint64_t rows() { return mRows; }
  uint64_t cols() { return mCols; }
  T* data() { return (T*) mPtr; }
  void set(uint64_t r, uint64_t c, T &val);
  void bit_transpose();
  void modp(uint64_t reducedTerms);
  GPUmatrix<T>& operator&=(T *rhs);
  void xor_async(T *rhs, cudaStream_t s);

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
  cudaMemcpy(mPtr + offset, &val, sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void GPUmatrix<T>::bit_transpose() {
  uint8_t *tpBuffer;
  cudaMalloc(&tpBuffer, mNBytes);
  dim3 nBlocks(32, 32);
  dim3 grid(mRows / 32, mCols / 32);
  bit_transposer<<<grid, nBlocks>>>(tpBuffer, (uint8_t*) mPtr);
  cudaDeviceSynchronize();
  uint64_t tpRows = mCols * 8 * sizeof(T);
  mCols = mRows / (8 * sizeof(T));
  mRows = tpRows;
}

template<typename T>
void GPUmatrix<T>::modp(uint64_t reducedTerms) {
  poly_mod_gpu<<<reducedTerms / 1024, 1024>>>((OTblock*) mPtr, mCols);
  cudaDeviceSynchronize();
  mCols = reducedTerms;
}

template<typename T>
GPUmatrix<T>& GPUmatrix<T>::operator&=(T *rhs) {
  and_single_gpu<<<mNBytes / 1024, 1024>>>((uint8_t*) mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
  cudaDeviceSynchronize();
}

template<typename T>
void GPUmatrix<T>::xor_async(T *rhs, cudaStream_t s) {
  xor_single_gpu<<<mNBytes / 1024, 1024, 0, s>>>((uint8_t*) mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
  cudaDeviceSynchronize();
}

#endif
