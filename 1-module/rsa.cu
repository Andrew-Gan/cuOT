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

uint32_t Rsa::modular_pow(uint32_t b, uint32_t e, uint32_t n) {
  uint64_t c = 1;
  for (int i = 0; i < e; i++) {
    c = (uint64_t)(b * c) % n;
  }
  return c;
}

std::pair<uint32_t, uint32_t> Rsa::getPublicKey() {
  return std::make_pair(e, n);
}

void Rsa::encrypt(uint32_t *m, size_t nBytes) {
  for (int i = 0; i < (nBytes-1) / 4 + 1; i++) {
    m[i] = modular_pow(m[i], e, n);
  }
}

void Rsa::decrypt(uint32_t *m, size_t nBytes) {
  for (int i = 0; i < (nBytes-1) / 4 + 1; i++) {
    m[i] = modular_pow(m[i], d, n);
  }
}
