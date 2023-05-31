#include "gpu_block.h"
#include "basic_op.h"
#include <iomanip>
#include <vector>
#include <mutex>

GPUBlock::GPUBlock() : GPUBlock(1024) {}

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

GPUBlock GPUBlock::operator*(const GPUBlock &rhs) {
  GPUBlock res(nBytes);
  // scalar multiplication
  if (nBytes > rhs.nBytes) {
    size_t numBlock = (rhs.nBytes - 1) / 1024 + 1;
    for (int i = 0; i < nBytes / rhs.nBytes; i++) {
      and_gpu<<<numBlock, 1024>>>(&res.data_d[i * rhs.nBytes], &data_d[i * rhs.nBytes], rhs.data_d, rhs.nBytes);
    }
    cudaDeviceSynchronize();
  }
  return res;
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

GPUBlock& GPUBlock::operator^=(const GPUBlock &rhs) {
  size_t numBlock = (nBytes - 1) / 1024 + 1;
  if (nBytes == rhs.nBytes)
    xor_gpu<<<numBlock, 1024>>>(data_d, data_d, rhs.data_d, nBytes);
  else
    xor_circular<<<numBlock, 1024>>>(data_d, data_d, rhs.data_d, rhs.nBytes, nBytes);
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

void GPUBlock::set(uint32_t val) {
  cudaMemset(data_d, 0, nBytes);
  cudaMemcpy(data_d, &val, sizeof(val), cudaMemcpyHostToDevice);
}

void GPUBlock::set(const uint8_t *val, size_t n) {
  cudaMemset(data_d, 0, nBytes);
  size_t min = nBytes < n ? nBytes : n;
  cudaMemcpy(data_d, val, min, cudaMemcpyHostToDevice);
}

GPUBlock GPUBlock::sum(size_t elemSize) {
  EventLog::start(SumNodes);
  GPUBlock res(*this);
  size_t numElem = nBytes / elemSize;
  sum_gpu<<<numElem / 2, elemSize>>>(res.data_d);
  cudaDeviceSynchronize();
  res.resize(elemSize);
  EventLog::end(SumNodes);
  return res;
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

