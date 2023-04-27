#include "rsa.h"

Rsa::Rsa() {
  e = 0x10001;
  d = 0x4bf16bb9;
  n = 0x79f99799;
}

Rsa::Rsa(uint32_t key_e, uint32_t key_n) {
  e = key_e;
  n = key_n;
}

std::pair<uint32_t, uint32_t> Rsa::getPublicKey() {
  return std::make_pair(e, n);
}

void Rsa::encrypt(uint32_t *m, size_t nBytes) {
  for (int i = 0; i < nBytes / 4; i++) {
    m[i] = (uint32_t) pow(m[i], e) % n;
  }
}

void Rsa::decrypt(uint32_t *m, size_t nBytes) {
  for (int i = 0; i < nBytes / 4; i++) {
    m[i] = (uint32_t) pow(m[i], d) % n;
  }
}
