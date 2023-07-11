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
  void resize(uint64_t r, uint64_t c);
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
  dim3 nBlocks(32, 32);
  dim3 grid(mCols * 128 / 32, mRows / 8 / 32);
  bit_transposer<<<grid, nBlocks>>>(tpBuffer, mPtr);
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
  uint64_t nBlk = (mNBytes + 1023) / 1024;
  and_single_gpu<<<nBlk, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
  cudaDeviceSynchronize();
}

template<typename T>
void GPUmatrix<T>::xor_async(T *rhs, cudaStream_t s) {
  uint64_t nBlk = (mNBytes + 1023) / 1024;
  xor_single_gpu<<<nBlk, 1024, 0, s>>>(mPtr, (uint8_t*) rhs, sizeof(T), mNBytes);
  cudaDeviceSynchronize();
}

#endif
