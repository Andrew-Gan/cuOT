#include <vector>
#include <algorithm>
#include "aes.h"
#include "aes_encrypt.h"
#include "aes_decrypt.h"
#include "aes_expand.h"
#include "utilsBox.h"

#define Nb 4
#define Nk 4
#define KEYSIZE_BITS 128

// state - array holding the intermediate results during decryption.
typedef uint8_t state_t[4][4];

Aes::~Aes() {
  if (encExpKey_d) cudaFree(encExpKey_d);
  if (decExpKey_d) cudaFree(decExpKey_d);
}

void Aes::init(uint8_t *key) {
  AES_ctx encExpKey;
  AES_ctx decExpKey;
  Aes::expand_encKey(encExpKey.roundKey, key);
  Aes::expand_decKey(decExpKey.roundKey, key);
  cudaError_t err = cudaMalloc(&encExpKey_d, sizeof(encExpKey.roundKey));
  if (err != cudaSuccess)
    fprintf(stderr, "Aes() enc: %s\n", cudaGetErrorString(err));
  cudaMemcpy(encExpKey_d, encExpKey.roundKey, sizeof(encExpKey.roundKey), cudaMemcpyHostToDevice);
  err = cudaMalloc(&decExpKey_d, sizeof(decExpKey.roundKey));
  if (err != cudaSuccess)
    fprintf(stderr, "Aes() dec: %s\n", cudaGetErrorString(err));
  cudaMemcpy(decExpKey_d, decExpKey.roundKey, sizeof(decExpKey.roundKey), cudaMemcpyHostToDevice);
}

void Aes::decrypt(GPUBlock &msg) {
  GPUBlock input(std::max(msg.nBytes, (uint64_t)1024));
  input.clear();
  cudaMemcpy(input.data_d, msg.data_d, msg.nBytes, cudaMemcpyDeviceToDevice);
  if (msg.nBytes < 1024) {
    msg = GPUBlock(1024);
  }
  dim3 grid(msg.nBytes / 4 / AES_BSIZE);
  aesDecrypt128<<<grid, AES_BSIZE>>>((uint32_t*) decExpKey_d, (uint32_t*) msg.data_d, (uint32_t*) input.data_d);
  cudaDeviceSynchronize();
}

void Aes::encrypt(GPUBlock &msg) {
  GPUBlock input(std::max(msg.nBytes, (uint64_t)1024));
  input.clear();
  cudaMemcpy(input.data_d, msg.data_d, msg.nBytes, cudaMemcpyDeviceToDevice);
  if (msg.nBytes < 1024) {
    msg = GPUBlock(1024);
  }
  dim3 grid(msg.nBytes / 4 / AES_BSIZE);
  aesEncrypt128<<<grid, AES_BSIZE>>>((uint32_t*) encExpKey_d, (uint32_t*) msg.data_d, (uint32_t*) input.data_d);
  cudaDeviceSynchronize();
}

void Aes::expand_async(OTBlock *interleaved, GPUBlock &separated, OTBlock *input_d, uint64_t width, int dir, cudaStream_t &s) {
  static int thread_per_aesblock = 4;
  uint64_t paddedBytes = (width / 2) * sizeof(*interleaved);
  if (paddedBytes % 1024 != 0)
    paddedBytes += 1024 - (paddedBytes % 1024);
  dim3 grid(paddedBytes * thread_per_aesblock / 16 / AES_BSIZE);
  aesExpand128<<<grid, AES_BSIZE, 0, s>>>((uint32_t*) encExpKey_d, interleaved, (uint32_t*) separated.data_d, (uint32_t*) input_d, dir, width);
}

static uint32_t myXor(uint32_t num1, uint32_t num2) {
	return num1 ^ num2;
}

static void single_step(std::vector<uint32_t> &expKey, uint32_t stepIdx){
	uint32_t num = 16;
	uint32_t idx = 16 * stepIdx;

	copy(expKey.begin()+(idx)-4, expKey.begin()+(idx),expKey.begin()+(idx));
	rotate(expKey.begin()+(idx), expKey.begin()+(idx)+1, expKey.begin()+(idx)+4);
	transform(expKey.begin()+idx, expKey.begin()+idx+4, expKey.begin()+idx, [](int n){return SBox[n];});
	expKey[idx] = expKey[idx] ^ Rcon[stepIdx-1];
	transform(expKey.begin()+(idx), expKey.begin()+(idx)+4, expKey.begin()+(idx)-num, expKey.begin()+(idx), myXor);
	for (int cnt = 0; cnt < 3; cnt++) {
		copy(expKey.begin()+(idx)+4*cnt, expKey.begin()+(idx)+4*(cnt+1),expKey.begin()+(idx)+(4*(cnt+1)));
		transform(expKey.begin()+(idx)+4*(cnt+1), expKey.begin()+(idx)+4*(cnt+2), expKey.begin()+(idx)-(num-4*(cnt+1)), expKey.begin()+(idx)+4*(cnt+1), myXor);
	}
}

static void _exp_func(std::vector<uint32_t> &keyArray, std::vector<uint32_t> &expKeyArray){
	copy(keyArray.begin(), keyArray.end(), expKeyArray.begin());
	for (int i = 1; i < 11; i++) {
		single_step(expKeyArray, i);
	}
}

static uint32_t _galois_prod(uint32_t a, uint32_t b) {

	if (a==0 || b==0) return 0;
	else {
		a = LogTable[a];
		b = LogTable[b];
		a = a+b;
		a = a % 255;
		a = ExpoTable[a];
		return a;
	}
}

static void _inv_mix_col(std::vector<unsigned> &temp){
	std::vector<unsigned> result(4);
	for(unsigned cnt=0; cnt<4; ++cnt){
		result[0] = _galois_prod(0x0e, temp[cnt*4]) ^ _galois_prod(0x0b, temp[cnt*4+1]) ^ _galois_prod(0x0d, temp[cnt*4+2]) ^ _galois_prod(0x09, temp[cnt*4+3]);
		result[1] = _galois_prod(0x09, temp[cnt*4]) ^ _galois_prod(0x0e, temp[cnt*4+1]) ^ _galois_prod(0x0b, temp[cnt*4+2]) ^ _galois_prod(0x0d, temp[cnt*4+3]);
		result[2] = _galois_prod(0x0d, temp[cnt*4]) ^ _galois_prod(0x09, temp[cnt*4+1]) ^ _galois_prod(0x0e, temp[cnt*4+2]) ^ _galois_prod(0x0b, temp[cnt*4+3]);
		result[3] = _galois_prod(0x0b, temp[cnt*4]) ^ _galois_prod(0x0d, temp[cnt*4+1]) ^ _galois_prod(0x09, temp[cnt*4+2]) ^ _galois_prod(0x0e, temp[cnt*4+3]);
		copy(result.begin(), result.end(), temp.begin()+(4*cnt));
	}
}

static void _inv_exp_func(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKey){
	std::vector<unsigned> temp(16);
	copy(expKey.begin(), expKey.begin()+16,invExpKey.end()-16);
	copy(expKey.end()-16, expKey.end(),invExpKey.begin());
	unsigned cycles = (expKey.size()!=240) ? 10 : 14;
	for (unsigned cnt=1; cnt<cycles; ++cnt){
		copy(expKey.end()-(16*cnt+16), expKey.end()-(16*cnt), temp.begin());
		_inv_mix_col(temp);
		copy(temp.begin(), temp.end(), invExpKey.begin()+(16*cnt));
	}
}

void Aes::expand_encKey(uint8_t *encExpKey, uint8_t *key){
  std::vector<uint32_t> keyArray(key, key + AES_KEYLEN);
	std::vector<uint32_t> expKeyArray(176);
  _exp_func(keyArray, expKeyArray);
  for (int cnt = 0; cnt < expKeyArray.size(); cnt++) {
    uint32_t val = expKeyArray[cnt];
    uint8_t *pc = reinterpret_cast<uint8_t*>(&val);
    encExpKey[cnt] = *pc;
  }
}

void Aes::expand_decKey(uint8_t *decExpKey, uint8_t *key){
  std::vector<uint32_t> keyArray(key, key + AES_KEYLEN);
  std::vector<uint32_t> expKeyArray(176);
	std::vector<uint32_t> invExpKeyArray(176);
  _exp_func(keyArray, expKeyArray);
  _inv_exp_func(expKeyArray, invExpKeyArray);
  for (int cnt = 0; cnt < invExpKeyArray.size(); cnt++) {
    uint32_t val = invExpKeyArray[cnt];
    uint8_t *pc = reinterpret_cast<uint8_t*>(&val);
    decExpKey[cnt] = *pc;
  }
}
