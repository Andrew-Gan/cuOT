#include "aes.h"
#include "aesEncrypt.h"
#include "aesDecrypt.h"
#include "utilsBox.h"

AesBlocks::AesBlocks() : AesBlocks(64) {}

AesBlocks::AesBlocks(size_t i_nBlock) {
  nBlock = i_nBlock;
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
  uint8_t key[AES_KEYLEN];
  for (int i = 0; i < AES_KEYLEN / 4; i++) {
    ((uint32_t*) key)[i] = rand();
  }
  AES_ctx exp_key;
  Aes::expand_key(exp_key.roundKey, key);
  cudaMalloc(&d_key, sizeof(exp_key.roundKey));
  cudaMemcpy(d_key, exp_key.roundKey, sizeof(exp_key.roundKey), cudaMemcpyHostToDevice);
}

Aes::Aes(uint8_t *newkey) {
  AES_ctx exp_key;
  Aes::expand_key(exp_key.roundKey, newkey);
  cudaMalloc(&d_key, sizeof(exp_key.roundKey));
  cudaMemcpy(d_key, exp_key.roundKey, sizeof(exp_key.roundKey), cudaMemcpyHostToDevice);
}

Aes::~Aes() {
  cudaFree(d_key);
}

void Aes::decrypt(AesBlocks *msg) {
  if (d_key == nullptr)
    return;
  uint8_t *d_buffer;
  cudaMalloc(&d_buffer, 16 * msg->nBlock);
  aesDecrypt128<<<4*msg->nBlock/AES_BSIZE, AES_BSIZE>>>((unsigned*) d_key, (unsigned*) d_buffer, (unsigned*) msg->d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(msg->d_data, d_buffer, 16 * msg->nBlock, cudaMemcpyDeviceToDevice);
  cudaFree(d_buffer);
}

void Aes::encrypt(AesBlocks *msg) {
  if (d_key == nullptr)
    return;
  uint8_t *d_buffer;
  cudaMalloc(&d_buffer, 16 * msg->nBlock);
  aesEncrypt128<<<4*msg->nBlock/AES_BSIZE, AES_BSIZE>>>((unsigned*) d_key, (unsigned*) d_buffer, (unsigned*) msg->d_data);
  cudaDeviceSynchronize();
  cudaMemcpy(msg->d_data, d_buffer, 16 * msg->nBlock, cudaMemcpyDeviceToDevice);
  cudaFree(d_buffer);
}

#define Nb 4
#define Nk 4

// state - array holding the intermediate results during decryption.
typedef uint8_t state_t[4][4];

// This function produces Nb(NUM_ROUNDS+1) round keys. The round keys are used in each round to decrypt the states.
void Aes::expand_key(uint8_t* roundKey, const uint8_t* Key) {
  unsigned i, j, k;
  uint8_t tempa[4]; // Used for the column/row operations

  // The first round key is the key itself.
  for (i = 0; i < Nk; ++i) {
    roundKey[(i * 4) + 0] = Key[(i * 4) + 0];
    roundKey[(i * 4) + 1] = Key[(i * 4) + 1];
    roundKey[(i * 4) + 2] = Key[(i * 4) + 2];
    roundKey[(i * 4) + 3] = Key[(i * 4) + 3];
  }

  // All other round keys are found from the previous round keys.
  for (i = Nk; i < Nb * (NUM_ROUNDS + 1); ++i) {
    {
      k = (i - 1) * 4;
      tempa[0]=roundKey[k + 0];
      tempa[1]=roundKey[k + 1];
      tempa[2]=roundKey[k + 2];
      tempa[3]=roundKey[k + 3];

    }

    if (i % Nk == 0)
    {
      // This function shifts the 4 bytes in a word to the left once.
      // [a0,a1,a2,a3] becomes [a1,a2,a3,a0]
      const uint8_t u8tmp = tempa[0];
      tempa[0] = tempa[1];
      tempa[1] = tempa[2];
      tempa[2] = tempa[3];
      tempa[3] = u8tmp;

      // SubWord() is a function that takes a four-byte input word and
      // applies the S-box to each of the four bytes to produce an output word.
      tempa[0] = SBox[tempa[0]];
      tempa[1] = SBox[tempa[1]];
      tempa[2] = SBox[tempa[2]];
      tempa[3] = SBox[tempa[3]];

      tempa[0] = tempa[0] ^ Rcon[i/Nk];
    }

    j = i * 4; k=(i - Nk) * 4;
    roundKey[j + 0] = roundKey[k + 0] ^ tempa[0];
    roundKey[j + 1] = roundKey[k + 1] ^ tempa[1];
    roundKey[j + 2] = roundKey[k + 2] ^ tempa[2];
    roundKey[j + 3] = roundKey[k + 3] ^ tempa[3];
  }
}
