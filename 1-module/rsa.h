#ifndef __RSA_H__
#define __RSA_H__

#include "util.h"
#include <utility>

class Rsa {
private:
  uint32_t e = 0, d = 0, n = 0;
  uint32_t modular_pow(uint32_t b, uint32_t e, uint32_t m);
public:
  Rsa();
  Rsa(uint32_t key_e, uint32_t key_n);
  std::pair<uint32_t, uint32_t> getPublicKey();
  void encrypt(uint32_t *m, size_t nBytes);
  void decrypt(uint32_t *m, size_t nBytes);
};

#endif
