#include "gpu_block.h"
#include "basic_op.h"
#include <iomanip>
#include <vector>
#include <mutex>

GPUBlock::GPUBlock() {
  nBytes = 0;
}

GPUBlock::GPUBlock(uint64_t n) {
  nBytes = n;
  cudaError_t err = cudaMalloc(&data_d, nBytes);
  if (err != cudaSuccess)
    fprintf(stderr, "GPUBlock(%lu): %s\n", nBytes, cudaGetErrorString(err));
}

GPUBlock::GPUBlock(const GPUBlock &blk) : GPUBlock(blk.nBytes) {
  cudaMemcpy(data_d, blk.data_d, nBytes, cudaMemcpyDeviceToDevice);
}

GPUBlock::~GPUBlock() {
  if (data_d != nullptr)
    cudaFree(data_d);
}

GPUBlock& GPUBlock::operator*=(const GPUBlock &rhs) {
  // scalar multiplication
  if (nBytes > rhs.nBytes) {
    uint64_t numBlock = (rhs.nBytes - 1) / 1024 + 1;
    for (int i = 0; i < nBytes / rhs.nBytes; i++) {
      and_gpu<<<numBlock, 1024>>>(&data_d[i * rhs.nBytes], rhs.data_d, rhs.nBytes);
    }
    cudaDeviceSynchronize();
  }
  return *this;
}

GPUBlock& GPUBlock::operator^=(const GPUBlock &rhs) {
  uint64_t numBlock = (nBytes - 1) / 1024 + 1;
  uint64_t minNBytes = std::min(nBytes, rhs.nBytes);
  xor_gpu<<<numBlock, 1024>>>(data_d, data_d, rhs.data_d, minNBytes);
  cudaDeviceSynchronize();
  return *this;
}

GPUBlock& GPUBlock::operator=(const GPUBlock &rhs) {
  if (nBytes != rhs.nBytes) {
    if (data_d != nullptr)
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
  OTBlock *nodes = new OTBlock[obj.nBytes];
  uint64_t numNode = obj.nBytes / sizeof(OTBlock);
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

void GPUBlock::set(const uint8_t *val, uint64_t n) {
  uint64_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d, val, min, cudaMemcpyHostToDevice);
}

void GPUBlock::set(const uint8_t *val, uint64_t n, uint64_t offset) {
  uint64_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d + offset, val, min, cudaMemcpyHostToDevice);
}

void GPUBlock::sum_async(uint64_t n, cudaStream_t stream) {
  uint64_t nBlocks = (( (n / 4) - 1) / 2048) + 1;
  xor_reduce_gpu<<<nBlocks, 1024, 4096, stream>>>((uint32_t*) data_d, n / 4);
}

void GPUBlock::xor_async(GPUBlock &rhs, cudaStream_t stream) {
  uint64_t numBlock = (nBytes - 1) / 1024 + 1;
  uint64_t minNBytes = std::min(nBytes, rhs.nBytes);
  xor_gpu<<<numBlock, 1024, 0, stream>>>(data_d, data_d, rhs.data_d, minNBytes);
}

void GPUBlock::copy_async(GPUBlock &rhs, cudaStream_t stream) {
  if (nBytes != rhs.nBytes) {
    if (data_d != nullptr)
      cudaFree(data_d);
    cudaError_t err = cudaMallocAsync(&data_d, rhs.nBytes, stream);
    if (err != cudaSuccess)
      fprintf(stderr, "operator=(GPUBlock): %s\n", cudaGetErrorString(err));
    nBytes = rhs.nBytes;
  }
  cudaMemcpyAsync(data_d, rhs.data_d, nBytes, cudaMemcpyDeviceToDevice, stream);
}

void GPUBlock::resize(uint64_t size) {
  uint8_t *newData;
  cudaError_t err = cudaMalloc(&newData, size);
  if (err != cudaSuccess)
    fprintf(stderr, "resize(%lu): %s\n", size, cudaGetErrorString(err));
  if (data_d != nullptr) {
    cudaMemcpy(newData, data_d, std::min(size, nBytes), cudaMemcpyDeviceToDevice);
    cudaFree(data_d);
  }
  data_d = newData;
  nBytes = size;
}

void GPUBlock::minCopy(GPUBlock &rhs) {
  uint64_t copySize = std::min(nBytes, rhs.nBytes);
  cudaMemcpy(data_d, rhs.data_d, copySize, cudaMemcpyDeviceToDevice);
}
