#include <random>
#include "base_ot.h"

uint32_t BaseOT::e = 0;
uint32_t BaseOT::n = 0;
uint32_t BaseOT::x[] = {0};
uint32_t BaseOT::v = 0;
uint32_t BaseOT::aesKey_enc = 0;
uint32_t *BaseOT::m[] = {nullptr};
InitStatus BaseOT::initStatus = noInit;
OTStatus BaseOT::otStatus = notReady;

__global__
void xor_gpu(uint32_t *out, uint32_t *data, uint32_t val) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  out[x] = data[x] ^ val;
}

void BaseOT::rsa_init() {
  BaseOT::e = 7;
  d = 3;
  BaseOT::n = 33;
}

uint32_t BaseOT::aesDecrypt(uint32_t m, uint32_t d, uint32_t n) {
  uint64_t blk[2] = {m};
  return blk[0];
}

uint32_t BaseOT::aesEncrypt(uint32_t m, uint32_t e, uint32_t n) {
  uint64_t blk[2] = {m};
  return blk[0];
}

BaseOT::BaseOT(Role role) {
  if (role == Sender) {
    rsa_init();
    BaseOT::x[0] = rand();
    BaseOT::x[1] = rand();
    initStatus = rsaInitDone;
    while(initStatus < aesInitDone);
    aesKey_s = (uint32_t) pow(BaseOT::aesKey_enc, d) % BaseOT::n;
  }
  else {
    while (initStatus < rsaInitDone);
    if (k_r == 0) {
      k_r = rand();
    }
    aesKey_r = rand();
    BaseOT::aesKey_enc = (uint32_t) pow(aesKey_r, BaseOT::e) % BaseOT::n;
    initStatus = aesInitDone;
  }
}

void BaseOT::ot_send(uint8_t *m0, uint8_t *m1, size_t nBytes) {
  if (role != Sender) {
    printf("Error not initalised as sender\n");
    return;
  }
  while(otStatus < vReady);
  k_s[0] = aesDecrypt(BaseOT::v ^ BaseOT::x[0], d, BaseOT::n);
  k_s[1] = aesDecrypt(BaseOT::v ^ BaseOT::x[1], d, BaseOT::n);
  m[0] = new uint32_t[nBytes / 4];
  m[1] = new uint32_t[nBytes / 4];
  xor_gpu<<<nBytes / 4 / 1024, 1024>>>(m[0], (uint32_t*) m0, k_s[0]);
  xor_gpu<<<nBytes / 4 / 1024, 1024>>>(m[1], (uint32_t*) m1, k_s[1]);
  cudaDeviceSynchronize();
  otStatus = mReady;
}

uint8_t* BaseOT::ot_recv(uint8_t b, size_t nBytes) {
  if (role != Recver) {
    printf("Error not initalised as recver\n");
    return nullptr;
  }
  v = x[b] ^ aesEncrypt(k_r, e, n);
  otStatus = vReady;
  while(otStatus < mReady);
  uint8_t *mb = new uint8_t[nBytes];
  xor_gpu<<<nBytes / 4 / 1024, 1024>>>((uint32_t*) mb, m[b], k_r);
  cudaDeviceSynchronize();
  delete[] m[0];
  delete[] m[1];
  otStatus = notReady;
  return mb;
}
