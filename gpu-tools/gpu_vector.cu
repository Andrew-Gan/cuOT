#include "gpu_ops.h"
#include "gpu_vector.h"

blk* GPUvector::data(uint64_t i) const {
  return GPUmatrix::data(0, i);
}

// nPartition: number of partitions to reduce to separate totals
// blkPerPart: number of blks in a partition
void GPUvector::sum(uint64_t nPartition, uint64_t blkPerPart) {
  uint64_t blockSize, nBlocks, mem;

  uint8_t *buffer;
  cudaMalloc(&buffer, this->mNBytes);

  uint64_t *in = (uint64_t*) buffer;
  uint64_t *out = (uint64_t*) this->mPtr;

  for (uint64_t remBlocks = blkPerPart; remBlocks > 1; remBlocks /= 1024) {
    std::swap(in, out);

    blockSize = std::min((uint64_t) 1024, remBlocks);
    nBlocks = nPartition * (remBlocks / blockSize);
    mem = blockSize * sizeof(uint64_t);
    xor_reduce<<<nBlocks, blockSize, mem>>>(out, in);
  }

  if (out != (uint64_t*) this->mPtr) {
    cudaMemcpy(this->mPtr, out, nPartition * sizeof(blk), cudaMemcpyDeviceToDevice);
  }

  cudaFree(buffer);
}

void GPUvector::xor_d(GPUvector &rhs, uint64_t offs) {
  uint64_t min = std::min(this->mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(this->mPtr, (uint8_t*) (rhs.data(offs)), min);
}
