#include "gpu_block.h"
#include "basic_op.h"

GPUBlock::GPUBlock() : GPUBlock(0) {}

GPUBlock::GPUBlock(size_t n) : nBytes(n) {
  cudaError_t err = cudaMalloc(&data_d, nBytes);
  if (err != cudaSuccess)
    fprintf(stderr, "GPUBlock(%u): %s\n", nBytes, cudaGetErrorString(err));
}

GPUBlock::GPUBlock(const GPUBlock &blk) : nBytes(blk.nBytes) {
  cudaError_t err = cudaMalloc(&data_d, blk.nBytes);
  if (err != cudaSuccess)
    fprintf(stderr, "GPUBlock(GPUBlock): %s\n", cudaGetErrorString(err));
  cudaMemcpy(data_d, blk.data_d, nBytes, cudaMemcpyDeviceToDevice);
}

GPUBlock::~GPUBlock() {
  cudaFree(data_d);
}

GPUBlock GPUBlock::operator^(const GPUBlock &rhs) {
  GPUBlock res(nBytes);
  size_t numBlock = (nBytes - 1) / 1024 + 1;
  if (nBytes == rhs.nBytes)
    xor_gpu<<<numBlock, 1024>>>(res.data_d, data_d, rhs.data_d, nBytes);
  else
    xor_circular<<<numBlock, 1024>>>(res.data_d, data_d, rhs.data_d, rhs.nBytes, nBytes);
  cudaDeviceSynchronize();
  return res;
}

GPUBlock& GPUBlock::operator=(const GPUBlock &rhs) {
  if (nBytes != rhs.nBytes) {
    cudaFree(data_d);
    cudaError_t err = cudaMalloc(&data_d, rhs.nBytes);
    if (err != cudaSuccess)
      fprintf(stderr, "operator=(GPUBlock): %s\n", cudaGetErrorString(err));
    nBytes = rhs.nBytes;
  }
  cudaMemcpy(data_d, rhs.data_d, nBytes, cudaMemcpyDeviceToDevice);
  return *this;
}

bool GPUBlock::operator==(const GPUBlock &rhs) {
  if (nBytes != rhs.nBytes)
    return false;
  uint8_t *left = new uint8_t[nBytes];
  uint8_t *right = new uint8_t[nBytes];
  cudaMemcpy(left, data_d, nBytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(right, rhs.data_d, nBytes, cudaMemcpyDeviceToHost);
  int cmp = memcmp(left, right, nBytes);
  delete[] left;
  delete[] right;
  return cmp == 0;
}

bool GPUBlock::operator!=(const GPUBlock &rhs) {
  return !(*this == rhs);
}

uint8_t& GPUBlock::operator[](int index) {
  return data_d[index];
}

void GPUBlock::set(uint32_t val) {
  cudaMemset(data_d, 0, nBytes);
  cudaMemcpy(data_d, &val, sizeof(val), cudaMemcpyHostToDevice);
}

void GPUBlock::set(const uint8_t *val, size_t n) {
  cudaMemset(data_d, 0, nBytes);
  size_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d, &val, min, cudaMemcpyHostToDevice);
}
