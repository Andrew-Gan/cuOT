#ifndef __RSA_H__
#define __RSA_H__

#include "util.h"

class Rsa {
private:
  uint32_t e = 0, d = 0, n = 0;
public:
  Rsa();
  Rsa(uint32_t key_e, uint32_t key_n);
  std::pair<uint32_t, uint32_t> getPublicKey();
  void encrypt(uint32_t *m, size_t nBytes);
  void decrypt(uint32_t *m, size_t nBytes);
};

#endif
