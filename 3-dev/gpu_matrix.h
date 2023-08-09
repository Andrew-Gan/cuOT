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
  uint64_t numThreadX = mCols * 8 * sizeof(OTblock);
  uint64_t numThreadY = mRows / 64;
  dim3 blocks(std::min((uint64_t) 32, numThreadX), std::min((uint64_t) 32, numThreadY));
  dim3 grid((numThreadX + 31) / 32, (numThreadY + 31) / 32);
  bit_transposer<<<grid, blocks>>>((uint64_t*) tpBuffer, (uint64_t*) mPtr);
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

template<typename T>
std::ostream& operator<<(std::ostream &os, const GPUmatrix<T> &obj) {
  T *nodes = new T[obj.rows() * obj.cols()];
  cudaMemcpy(nodes, obj.data(), obj.size_bytes(), cudaMemcpyDeviceToHost);
  for (uint64_t r = 0; r < obj.rows(); r++) {
    for (uint64_t c = 0; c < obj.cols(); c++) {
      uint64_t offs = r * obj.cols() + c;
      os << nodes[offs] << " ";
    }
    os << std::endl;
  }
  delete[] nodes;
  return os;
}

#endif
