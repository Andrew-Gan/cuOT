#include "rsa.h"

Rsa::Rsa() {
  e = 0x10001;
  d = 0x32741441;
  n = 0x73fe2131;
}

Rsa::Rsa(uint32_t key_e, uint32_t key_n) {
  e = key_e;
  n = key_n;
}

__global__
void modular_pow(uint32_t *b, uint32_t e, uint32_t n) {
  uint64_t c = 1;
  for (int i = 0; i < e; i++) {
    c = (uint64_t)(*b * c) % n;
  }
  *b = c;
}

void Rsa::encrypt(GPUBlock m) {
  EventLog::start(RsaEncrypt);
  uint32_t *casted = (uint32_t*) m.data_d;
  for (int i = 0; i < (m.nBytes-1) / 4 + 1; i++) {
    modular_pow<<<1, 1>>>(&casted[i], e, n);
    cudaDeviceSynchronize();
  }
  EventLog::end(RsaEncrypt);
}

void Rsa::decrypt(GPUBlock m) {
  EventLog::start(RsaDecrypt);
  uint32_t *casted = (uint32_t*) m.data_d;
  for (int i = 0; i < (m.nBytes-1) / 4 + 1; i++) {
    modular_pow<<<1, 1>>>(&casted[i], d, n);
    cudaDeviceSynchronize();
  }
  EventLog::end(RsaDecrypt);
}
