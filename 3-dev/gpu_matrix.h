#ifndef __GPU_MATRIX_H__
#define __GPU_MATRIX_H__

#include "gpu_block.h"

template<typename T>
class GPUMatrix {
public:
  uint64_t rows, cols;
  GPUBlock block;
  GPUMatrix() : rows(0), cols(0) {}
  GPUMatrix(uint64_t r, uint64_t c);
  T& at(size_t r, size_t c);
  T* get_dptr();
};

template<typename T>
GPUMatrix<T>::GPUMatrix(uint64_t r, uint64_t c) : rows(r), cols(c) {
  block.resize(rows * cols * sizeof(T));
  block.clear();
}

template<typename T>
T& GPUMatrix<T>::at(size_t r, size_t c) {
  T *ref = ((T*) block.data_d) + r * cols + c;
  return *ref;
}

template<typename T>
T* GPUMatrix<T>::get_dptr() {
  return (T*) block.data_d;
}

#endif
