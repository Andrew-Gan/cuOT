#include <vector>
#include <algorithm>
#include "pprf.h"
#include "aes_op.h"
#include "utilsBox.h"
#include "gpu_tests.h"

#define Nb 4
#define Nk 4
#define KEYSIZE_BITS 128

// state - array holding the intermediate results during decryption.
typedef uint8_t state_t[4][4];

__constant__ uint8_t keyLeft[11*AES_KEYLEN];
__constant__ uint8_t keyRight[11*AES_KEYLEN];

Aes::Aes(void *leftUnexpSeed, void *rightUnexpSeed) {
  AES_ctx leftExpKey;
  Aes::expand_encKey(leftExpKey.roundKey, (uint8_t*)leftUnexpSeed);
  cudaMemcpyToSymbol(keyLeft, leftExpKey.roundKey, sizeof(keyLeft));
  cudaGetSymbolAddress((void**)&keyL, keyLeft);
  if (rightUnexpSeed != nullptr) {
    AES_ctx rightExpKey;
    Aes::expand_encKey(rightExpKey.roundKey, (uint8_t*)rightUnexpSeed);
    cudaMemcpyToSymbol(keyRight, rightExpKey.roundKey, sizeof(keyRight));
    cudaGetSymbolAddress((void**)&keyR, keyRight);
    hasBothKeys = true;
  }
}

void Aes::encrypt(Mat &data) {
  uint64_t grid = (4 * data.size() + AES_BSIZE - 1) / AES_BSIZE;
  aesEncrypt128<<<grid, AES_BSIZE>>>((uint32_t*)keyL, (uint32_t*)data.data());
  cudaDeviceSynchronize();
}

void Aes::expand(Mat &interleaved_in, Mat &interleaved_out, Mat &separated, uint64_t inWidth) {
  if (!hasBothKeys) {
    throw std::runtime_error("Aes initialised with only one key");
  }
  dim3 grid((4*inWidth+(AES_BSIZE-1)) / AES_BSIZE, 2);
  aesExpand128<<<grid, AES_BSIZE>>>(keyL, keyR, interleaved_in.data(),
    interleaved_out.data(), separated.data(), inWidth);\
  cudaDeviceSynchronize();
}

uint32_t myXor(uint32_t num1, uint32_t num2) {
  return num1 ^ num2;
}

void Aes::single_step(std::vector<uint32_t> &expKey, uint32_t stepIdx) {
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

void Aes::exp_func(std::vector<uint32_t> &keyArray, std::vector<uint32_t> &expKeyArray) {
  copy(keyArray.begin(), keyArray.end(), expKeyArray.begin());
  for (int i = 1; i < 11; i++) {
    Aes::single_step(expKeyArray, i);
  }
}

uint32_t Aes::galois_prod(uint32_t a, uint32_t b) {

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

void Aes::inv_mix_col(std::vector<unsigned> &temp) {
  std::vector<unsigned> result(4);
  for(unsigned cnt=0; cnt<4; ++cnt){
    result[0] = Aes::galois_prod(0x0e, temp[cnt*4]) ^ Aes::galois_prod(0x0b, temp[cnt*4+1]) ^ Aes::galois_prod(0x0d, temp[cnt*4+2]) ^ Aes::galois_prod(0x09, temp[cnt*4+3]);
    result[1] = Aes::galois_prod(0x09, temp[cnt*4]) ^ Aes::galois_prod(0x0e, temp[cnt*4+1]) ^ Aes::galois_prod(0x0b, temp[cnt*4+2]) ^ Aes::galois_prod(0x0d, temp[cnt*4+3]);
    result[2] = Aes::galois_prod(0x0d, temp[cnt*4]) ^ Aes::galois_prod(0x09, temp[cnt*4+1]) ^ Aes::galois_prod(0x0e, temp[cnt*4+2]) ^ Aes::galois_prod(0x0b, temp[cnt*4+3]);
    result[3] = Aes::galois_prod(0x0b, temp[cnt*4]) ^ Aes::galois_prod(0x0d, temp[cnt*4+1]) ^ Aes::galois_prod(0x09, temp[cnt*4+2]) ^ Aes::galois_prod(0x0e, temp[cnt*4+3]);
    copy(result.begin(), result.end(), temp.begin()+(4*cnt));
  }
}

void Aes::inv_exp_func(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKey) {
  std::vector<unsigned> temp(16);
  copy(expKey.begin(), expKey.begin()+16,invExpKey.end()-16);
  copy(expKey.end()-16, expKey.end(),invExpKey.begin());
  unsigned cycles = (expKey.size()!=240) ? 10 : 14;
  for (unsigned cnt=1; cnt<cycles; ++cnt){
    copy(expKey.end()-(16*cnt+16), expKey.end()-(16*cnt), temp.begin());
    Aes::inv_mix_col(temp);
    copy(temp.begin(), temp.end(), invExpKey.begin()+(16*cnt));
  }
}

void Aes::expand_encKey(uint8_t *encExpKey, uint8_t *key){
  std::vector<uint32_t> keyArray(key, key + AES_KEYLEN);
  std::vector<uint32_t> expKeyArray(176);
  Aes::exp_func(keyArray, expKeyArray);
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
  Aes::exp_func(keyArray, expKeyArray);
  Aes::inv_exp_func(expKeyArray, invExpKeyArray);
  for (int cnt = 0; cnt < invExpKeyArray.size(); cnt++) {
    uint32_t val = invExpKeyArray[cnt];
    uint8_t *pc = reinterpret_cast<uint8_t*>(&val);
    decExpKey[cnt] = *pc;
  }
}
