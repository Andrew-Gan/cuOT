#ifndef __RSA_H__
#define __RSA_H__

#include <utility>
#include "util.h"
#include "gpu_block.h"

class Rsa {
private:
  uint32_t d;
public:
  uint32_t e, n;
  Rsa();
  Rsa(uint32_t key_e, uint32_t key_n);
  void encrypt(GPUBlock m);
  void decrypt(GPUBlock m);
};

#endif
