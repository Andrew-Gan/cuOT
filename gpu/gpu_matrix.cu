#include <iomanip>
#include <bitset>
#include "gpu_ops.h"
#include "gpu_matrix.h"
#include <stdexcept>
#include "gpu_tests.h"

Mat::Mat(std::vector<uint64_t> newDim) : GPUdata(listToSize(newDim)*sizeof(blk)) {
  mDim = newDim;
  buffer_adjust();
}

Mat::Mat(const Mat &other) : GPUdata(other) {
  mDim = other.mDim;
  buffer_adjust();
}

Mat::~Mat() {
  if (bufferSize != 0) {
    cudaSetDevice(mDevice);
    cudaError_t err = cudaFree(buffer);
    if (err != cudaSuccess) {
      std::cout << cudaGetErrorString(err) << std::endl;
    }
  }
}

void Mat::buffer_adjust() {
  if (bufferSize > 0 && bufferSize < mNBytes) {
    cudaSetDevice(mDevice);
    cudaFree(buffer);
    bufferSize = 0;
  }
  if (bufferSize == 0) {
    cudaSetDevice(mDevice);
    cudaMalloc(&buffer, mNBytes);
    bufferSize = mNBytes;
  }
}

uint64_t Mat::dim(uint32_t i) const {
  if (mDim.size() == 0)
    return 0;
  if (i >= mDim.size())
    throw std::invalid_argument("Requested dim exceeds matrix dim\n");
  return mDim.at(i);
}

uint64_t Mat::listToSize(std::vector<uint64_t> dim) {
  if (dim.size() == 0)
    return 0;
  uint64_t size = 1;
  for (const uint64_t &i : dim)
    size *= i;
  return size;
}

uint64_t Mat::listToOffset(std::vector<uint64_t> pos) const {
  if (pos.size() != mDim.size())
    throw std::invalid_argument("Matrix dim and pos len mismatch\n");

  uint64_t offs = 0;
  for (int i = 0; i < pos.size() - 1; i++)
    offs += pos.at(i) * mDim.at(i+1);
  offs += pos.back();
  return offs;
}

blk* Mat::data(std::vector<uint64_t> pos) const {
  if (pos.size() != mDim.size())
    throw std::invalid_argument("Matrix dim and pos dim mismatch\n");

  for (int i = 0; i < pos.size(); i++) {
    if (pos.at(i) >= mDim.at(i)) {
      std::ostringstream msg;
      msg << "Mat::data exceeded dim " << i << ", accessing " << pos.at(i) << " when max is " << mDim.at(i) << std::endl;
      throw std::invalid_argument(msg.str());
    }
  }
  
  return (blk*)mPtr + listToOffset(pos);
}

void Mat::set(blk &val, std::vector<uint64_t> pos) {
  uint64_t offset = listToOffset(pos);
  cudaMemcpy((blk*)mPtr + offset, &val, sizeof(blk), cudaMemcpyHostToDevice);
}

void Mat::resize(std::vector<uint64_t> newDim) {
  GPUdata::resize(listToSize(newDim)*sizeof(blk));
  buffer_adjust();
  mDim = newDim;
}

__global__
void bit_transpose_kernel(uint8_t *out, uint8_t *in, dim3 matDimBytes, uint64_t start, uint64_t end) {
  uint64_t uid = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef USE_IMPROVED_BIT_TRANSPOSE
  extern __shared__ uint64_t tile[];
  uint64_t i = uid / (end - start);
  uint64_t j = uid % (end - start);

  uint64_t tmp =
    ( uint64_t( in[   i * 8       * matDimBytes.x + start + j ] ) << 0  ) |
    ( uint64_t( in[ ( i * 8 + 1 ) * matDimBytes.x + start + j ] ) << 8  ) |
    ( uint64_t( in[ ( i * 8 + 2 ) * matDimBytes.x + start + j ] ) << 16 ) |
    ( uint64_t( in[ ( i * 8 + 3 ) * matDimBytes.x + start + j ] ) << 24 ) |
    ( uint64_t( in[ ( i * 8 + 4 ) * matDimBytes.x + start + j ] ) << 32 ) |
    ( uint64_t( in[ ( i * 8 + 5 ) * matDimBytes.x + start + j ] ) << 40 ) |
    ( uint64_t( in[ ( i * 8 + 6 ) * matDimBytes.x + start + j ] ) << 48 ) |
    ( uint64_t( in[ ( i * 8 + 7 ) * matDimBytes.x + start + j ] ) << 56 );
  tmp =
    (tmp & 0x8040201008040201LL) |
    ((tmp & 0x0080402010080402LL) <<  7) |
    ((tmp & 0x0000804020100804LL) << 14) |
    ((tmp & 0x0000008040201008LL) << 21) |
    ((tmp & 0x0000000080402010LL) << 28) |
    ((tmp & 0x0000000000804020LL) << 35) |
    ((tmp & 0x0000000000008040LL) << 42) |
    ((tmp & 0x0000000000000080LL) << 49) |
    ((tmp >>  7) & 0x0080402010080402LL) |
    ((tmp >> 14) & 0x0000804020100804LL) |
    ((tmp >> 21) & 0x0000008040201008LL) |
    ((tmp >> 28) & 0x0000000080402010LL) |
    ((tmp >> 35) & 0x0000000000804020LL) |
    ((tmp >> 42) & 0x0000000000008040LL) |
    ((tmp >> 49) & 0x0000000000000080LL);
  tile[i * matDimBytes.x + j] = tmp;
  __syncthreads();

  i = uid / matDimBytes.y;
  j = uid % matDimBytes.y;
  tmp = tile[j * matDimBytes.x + i];
  out[ ( i * 8 ) * matDimBytes.y + j ]     = uint8_t(tmp);
  out[ ( i * 8 + 1 ) * matDimBytes.y + j ] = uint8_t(tmp >> 8);
  out[ ( i * 8 + 2 ) * matDimBytes.y + j ] = uint8_t(tmp >> 16);
  out[ ( i * 8 + 3 ) * matDimBytes.y + j ] = uint8_t(tmp >> 24);
  out[ ( i * 8 + 4 ) * matDimBytes.y + j ] = uint8_t(tmp >> 32);
  out[ ( i * 8 + 5 ) * matDimBytes.y + j ] = uint8_t(tmp >> 40);
  out[ ( i * 8 + 6 ) * matDimBytes.y + j ] = uint8_t(tmp >> 48);
  out[ ( i * 8 + 7 ) * matDimBytes.y + j ] = uint8_t(tmp >> 56);
#else
  uint64_t r_des = uid / (matDimBytes.y);
  uint64_t c_des_byte = uid % (matDimBytes.y);

  for (int j = 0; j < 8; j++) {
    uint64_t c_des_bit = 8 * c_des_byte + j;
    uint64_t r_src = c_des_bit;
    uint64_t c_src_bit = start + r_des;
    uint64_t tmp = in[r_src * matDimBytes.x + (c_src_bit / 8)];
    tmp &= 1 << (c_src_bit % 8);
    out[r_des * matDimBytes.y / 8 + c_des_bit / 8] |= tmp;
  }
#endif // USE_IMPROVED_BIT_TRANSPOSE
}

void Mat::bit_transpose(uint64_t startColBit, uint64_t endColBit) {
  uint64_t rowBytes = dim(0) / 8;
  uint64_t colBytes = dim(1) * sizeof(blk);
  dim3 matDimBytes(colBytes, rowBytes);

  if (endColBit == 0) endColBit = 8 * colBytes;
  if (mDim.size() != 2)
    throw std::invalid_argument("Mat::bit_transpose only supports 2D matrix\n");

#ifdef USE_IMPROVED_BIT_TRANSPOSE
  uint64_t nThread = rowBytes * (endColBit - startColBit) / 8;
  uint64_t block = std::min(1024UL, nThread);
  uint64_t grid = (nThread + block - 1) / block;
  uint64_t shMem = nThread * sizeof(uint64_t);
  bit_transpose_kernel<<<grid, block, shMem>>>(buffer, mPtr, matDimBytes, startColBit/8, endColBit/8);
#else
  uint64_t nThread = (endColBit - startColBit) * rowBytes;
  uint64_t block = std::min(1024UL, nThread);
  uint64_t grid = (nThread + block - 1) / block;
  bit_transpose_kernel<<<grid, block>>>(buffer, mPtr, matDimBytes, startColBit/8, endColBit/8);
#endif // USE_IMPROVED_BIT_TRANSPOSE

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
    printf("%s\n", cudaGetErrorString(err));
  std::swap(buffer, mPtr);

  mDim.at(1) = std::max(1UL, rowBytes / sizeof(blk));
  mDim.at(0) = endColBit - startColBit;
  mNBytes = listToSize(mDim) * sizeof(blk);
}

__global__
void mod_helper(uint8_t *a, uint8_t *b, uint64_t n, uint64_t rowBytes) {
  uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t numX = gridDim.x * blockDim.x;
  uint64_t y = blockIdx.y;
  if (x < n) a[y * numX + x] ^= b[y * rowBytes + x];
}

void Mat::modp(uint64_t reducedCol) {
  if (mDim.size() > 2)
    throw std::invalid_argument("Mat::modp only 1D or 2D matrix supported\n");

  uint64_t col = mDim.back();
  uint64_t threads = reducedCol * sizeof(blk);
  uint64_t block = std::min(threads, 1024lu);
  uint64_t rows = mDim.size() == 2 ? mDim.front() : 1;
  dim3 grid = dim3((threads + block - 1) / block, rows);

  for (uint64_t i = 0; i < col / reducedCol; i++)
    gpu_xor<<<grid, block>>>(buffer, mPtr+i*threads, threads, col*sizeof(blk));

  std::swap(mPtr, buffer);
  std::vector<uint64_t> newDim = mDim;
  newDim.back() = reducedCol;
  resize(newDim);
}

void Mat::xor_scalar(blk *rhs, uint64_t numBlock) {
  uint64_t bytesToXor = numBlock == 0 ? mNBytes : 16 * numBlock;
  uint64_t nBlock = (bytesToXor + 1023) / 1024;
  xor_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), bytesToXor);
}

#ifdef USE_COALESCED_NODE_SUMMATION
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
__device__
void warp_reduce(uint64_t *sdata, uint64_t tid) {
  if (blockDim.x >= 64 && tid < 32) sdata[tid] ^= sdata[tid + 32];
  if (blockDim.x >= 32 && tid < 16) sdata[tid] ^= sdata[tid + 16];
  if (blockDim.x >= 16 && tid < 8) sdata[tid] ^= sdata[tid + 8];
  if (blockDim.x >= 8 && tid < 4) sdata[tid] ^= sdata[tid + 4];
  if (blockDim.x >= 4 && tid < 2) sdata[tid] ^= sdata[tid + 2];
}

__global__
void xor_reduce(uint64_t *out, uint64_t *in) {
  extern __shared__ uint64_t sdata[];
  uint64_t tid = threadIdx.x;
  uint64_t start = blockIdx.x * (2 * blockDim.x);

  sdata[tid] = in[start + tid] ^ in[start + tid + blockDim.x];
  __syncthreads();
  if (blockDim.x == 1024 && tid < 512) sdata[tid] ^= sdata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256) sdata[tid] ^= sdata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128) sdata[tid] ^= sdata[tid + 128];
  __syncthreads();
  if (blockDim.x >= 128 && tid < 64) sdata[tid] ^= sdata[tid + 64];
  __syncthreads();
  if (tid < 32) warp_reduce(sdata, tid);
  if (tid < 2) out[2 * blockIdx.x + tid] = sdata[tid];
}
#else
__global__ void summator(blk *out, blk *in) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = 0; j < 4; j++)
    out[i].data_32[j] = in[2*i].data_32[j] ^ in[2*i+1].data_32[j];
}

__global__ void gatherer(blk *out, blk *in, uint64_t blkPerPart) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  out[i] = in[i * blkPerPart];
}
#endif // USE_COALESCED_NODE_SUMMATION

void Mat::sum(uint64_t nPartition, uint64_t blkPerPart) {
#ifdef USE_COALESCED_NODE_SUMMATION
  uint64_t *in = (uint64_t*)buffer;
  uint64_t *out = (uint64_t*)this->mPtr;
  for (uint64_t nThread = blkPerPart; nThread > 1; nThread /= 1024) {
    std::swap(in, out);
    uint64_t block = std::min(1024UL, nThread);
    uint64_t grid = nPartition * (nThread / block);
    uint64_t mem = block * sizeof(uint64_t);
    xor_reduce<<<grid, block, mem>>>(out, in);
  }
#else
  blk *in = (blk*)this->mPtr;
  blk *out = (blk*)buffer;
  for (int p = 0; p < nPartition; p++) {
    for (uint64_t nThread = blkPerPart / 2; nThread >= 1; nThread /= 2) {
      uint64_t block = std::min(1024UL, nThread);
      uint64_t grid = nThread / block;
      summator<<<grid, block>>>(out + p * blkPerPart, in + p * blkPerPart);
      std::swap(in, out);
    }
  }
  gatherer<<<1, nPartition>>>(out, in, blkPerPart);
#endif

  mPtr = (uint8_t*)out;
  buffer = (uint8_t*)in;
}

void Mat::xor_d(Mat &rhs, uint64_t offs) {
  uint64_t min = std::min(this->mNBytes, rhs.size_bytes());
  uint64_t nBlock = (min + 1023) / 1024;
  gpu_xor<<<nBlock, 1024>>>(this->mPtr, (uint8_t*)(rhs.data() + offs), min);
}

Mat& Mat::operator&=(blk *rhs) {
  uint64_t nBlock = (mNBytes + 1023) / 1024;
  and_single<<<nBlock, 1024>>>(mPtr, (uint8_t*) rhs, sizeof(blk), mNBytes);
  return *this;
}

Mat& Mat::operator=(Mat &other) {
  GPUdata::operator=(other);
  buffer_adjust();
  return *this;
}

std::ostream& operator<<(std::ostream &os, Mat &obj) {
  if (obj.dims().size() > 2)
    throw std::invalid_argument("Mat::operator<< only 1D or 2D matrix supported\n");
  blk *tmp = new blk[obj.size_bytes() / sizeof(blk)];
  uint64_t rows = obj.dims().size() == 2 ? obj.dim(0) : 1;
  uint64_t cols = obj.dims().size() == 2 ? obj.dim(1) : obj.dim(0);
  cudaMemcpy(tmp, obj.data(), obj.size_bytes(), cudaMemcpyDeviceToHost);
  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      blk *val = tmp+i*cols+j;
      for (int i = 0; i < 1; i++) {
        os << std::setw(8) << std::setfill('0') << std::hex << val->data_32[i];
      }
      os << " ";
    }
    os << std::endl;
  }
  os << std::dec;
  delete[] tmp;
  return os;
}
