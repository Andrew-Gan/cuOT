#include "gpu_block.h"
#include "basic_op.h"
#include <iomanip>
#include <vector>
#include <mutex>

GPUBlock::GPUBlock() {
  nBytes = 0;
}

GPUBlock::GPUBlock(size_t n) {
  nBytes = n;
  cudaError_t err = cudaMalloc(&data_d, nBytes);
  if (err != cudaSuccess)
    fprintf(stderr, "GPUBlock(%lu): %s\n", nBytes, cudaGetErrorString(err));
}

GPUBlock::GPUBlock(const GPUBlock &blk) : GPUBlock(blk.nBytes) {
  cudaMemcpy(data_d, blk.data_d, nBytes, cudaMemcpyDeviceToDevice);
}

GPUBlock::~GPUBlock() {
  cudaFree(data_d);
}

GPUBlock& GPUBlock::operator*=(const GPUBlock &rhs) {
  // scalar multiplication
  if (nBytes > rhs.nBytes) {
    size_t numBlock = (rhs.nBytes - 1) / 1024 + 1;
    for (int i = 0; i < nBytes / rhs.nBytes; i++) {
      and_gpu<<<numBlock, 1024>>>(&data_d[i * rhs.nBytes], rhs.data_d, rhs.nBytes);
    }
    cudaDeviceSynchronize();
  }
  return *this;
}

GPUBlock& GPUBlock::operator^=(const GPUBlock &rhs) {
  size_t numBlock = (nBytes - 1) / 1024 + 1;
  size_t minNBytes = std::min(nBytes, rhs.nBytes);
  xor_gpu<<<numBlock, 1024>>>(data_d, data_d, rhs.data_d, minNBytes);
  cudaDeviceSynchronize();
  return *this;
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

std::ostream& operator<<(std::ostream &os, const GPUBlock &obj) {
  static std::mutex mtx;

  mtx.lock();
  TreeNode *nodes = new TreeNode[obj.nBytes];
  size_t numNode = obj.nBytes / sizeof(TreeNode);
  cudaMemcpy(nodes, obj.data_d, obj.nBytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < numNode; i += 16) {
    for (int j = i; j < numNode && j < (i + 16); j++) {
      os << std::setfill('0') << std::setw(2) << std::hex << +nodes[j].data[0] << " ";
    }
    os << std::endl;
  }
  delete[] nodes;
  mtx.unlock();

  return os;
}

void GPUBlock::clear() {
  cudaMemset(data_d, 0, nBytes);
}

void GPUBlock::set(uint64_t val) {
  cudaMemcpy(data_d, &val, sizeof(val), cudaMemcpyHostToDevice);
}

void GPUBlock::set(const uint8_t *val, size_t n) {
  size_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d, val, min, cudaMemcpyHostToDevice);
}

void GPUBlock::set(const uint8_t *val, size_t n, size_t offset) {
  size_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d + offset, val, min, cudaMemcpyHostToDevice);
}

void GPUBlock::sum_async(size_t elemSize) {
  size_t numLL = nBytes / sizeof(uint64_t);
  size_t sharedMemsize = 1024 * sizeof(uint64_t);
  sum_gpu<<<numLL / 2048, 1024, sharedMemsize>>>((uint64_t*) data_d, numLL);
}

void GPUBlock::resize(size_t size) {
  uint8_t *newData;
  cudaMalloc(&newData, size);
  cudaMemcpy(newData, data_d, std::min(size, nBytes), cudaMemcpyDeviceToDevice);
  cudaFree(data_d);
  data_d = newData;
  nBytes = size;
}

void GPUBlock::append(GPUBlock &rhs) {
  uint8_t *appendedData;
  cudaMalloc(&appendedData, nBytes + rhs.nBytes);
  cudaMemcpy(appendedData, data_d, nBytes, cudaMemcpyDeviceToDevice);
  cudaMemcpy(appendedData + nBytes, rhs.data_d, rhs.nBytes, cudaMemcpyDeviceToDevice);
  cudaFree(data_d);
  data_d = appendedData;
  nBytes += rhs.nBytes;
}

void GPUBlock::minCopy(GPUBlock &rhs) {
  size_t copySize = std::min(nBytes, rhs.nBytes);
  cudaMemcpy(data_d, rhs.data_d, copySize, cudaMemcpyDeviceToDevice);
}
