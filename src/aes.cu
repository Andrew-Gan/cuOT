#include "aes.h"
#include "aesEncrypt.h"
#include "aesDecrypt.h"

AesBlocks::AesBlocks() : AesBlocks(1) {}

AesBlocks::AesBlocks(size_t nBlock) {
  cudaMalloc(&d_data, 16 * nBlock);
}

AesBlocks::~AesBlocks() {
  cudaFree(d_data);
}

__global__
static void xor_pairwise(uint8_t *d_out, uint8_t *d_in0, uint8_t *d_in1) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  d_out[x] = d_in0[x] ^ d_in1[x];
}

__global__
static void xor_uneven(uint8_t *d_out, uint8_t *d_in, uint8_t *d_rep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  d_out[x] = d_in[x] ^ d_rep[x%16];
}

AesBlocks AesBlocks::operator^(const AesBlocks &rhs) {
  AesBlocks res(nBlock);
  if (nBlock == rhs.nBlock)
    xor_pairwise<<<nBlock, 16>>>(res.d_data, d_data, rhs.d_data);
  else if (rhs.nBlock == 1)
    xor_uneven<<<nBlock, 16>>>(res.d_data, d_data, rhs.d_data);
  return res;
}

AesBlocks AesBlocks::operator=(uint32_t rhs) {
  cudaMemcpy(d_data, &rhs, sizeof(rhs), cudaMemcpyHostToDevice);
  return *this;
}

AesBlocks AesBlocks::operator=(const AesBlocks &rhs) {
  if (nBlock != rhs.nBlock) {
    cudaFree(d_data);
    cudaMalloc(&d_data, 16 * rhs.nBlock);
    nBlock = rhs.nBlock;
  }
  cudaMemcpy(d_data, rhs.d_data, 16 * nBlock, cudaMemcpyDeviceToDevice);
  return *this;
}

Aes::Aes() {
  cudaMalloc(&d_key, AES_BLOCKLEN);
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
  curandSetPseudoRandomGeneratorSeed(prng, clock());
  curandGenerateUniform(prng, (float*) d_key, AES_BLOCKLEN / 4);
}

Aes::Aes(uint8_t *newkey) {
  cudaMalloc(&d_key, AES_BLOCKLEN);
  cudaMemcpy(d_key, newkey, AES_BLOCKLEN, cudaMemcpyHostToDevice);
}

Aes::~Aes() {
  cudaFree(d_key);
}

void Aes::decrypt(AesBlocks msg) {
  if (d_key == nullptr)
    return;
  uint8_t *d_buffer;
  cudaMalloc(&d_buffer, 16 * msg.nBlock);
  aesDecrypt128<<<4 * msg.nBlock / AES_BSIZE, AES_BSIZE>>>((unsigned*) d_key, (unsigned*) d_buffer, (unsigned*) msg.d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(msg.d_data, d_buffer, 16 * msg.nBlock, cudaMemcpyDeviceToDevice);
  cudaFree(d_buffer);
}

void Aes::encrypt(AesBlocks msg) {
  if (d_key == nullptr)
    return;
  uint8_t *d_buffer;
  cudaMalloc(&d_buffer, 16 * msg.nBlock);
  aesEncrypt128<<<4 * msg.nBlock / AES_BSIZE, AES_BSIZE>>>((unsigned*) d_key, (unsigned*) d_buffer, (unsigned*) msg.d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(msg.d_data, d_buffer, 16 * msg.nBlock, cudaMemcpyDeviceToDevice);
  cudaFree(d_buffer);
}
