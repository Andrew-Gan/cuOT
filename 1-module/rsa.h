#ifndef __RSA_H__
#define __RSA_H__

#include "util.h"
#include "gpu_block.h"

#define RSAKEY_BITLEN 1024

class Rsa {
private:
  GPUBlock p(RSAKEY_BITLEN / 2);
  GPUBlock q(RSAKEY_BITLEN / 2);
  GPUBlock d_p(RSAKEY_BITLEN / 2);
  GPUBlock d_q(RSAKEY_BITLEN / 2);
  GPUBlock q_inv(RSAKEY_BITLEN / 2);

public:
  GPUBlock e(RSAKEY_BITLEN / 2);
  GPUBlock n(RSAKEY_BITLEN);
  Rsa();
  Rsa(uint1024_t key_e, uint1024_t key_n);
  void encrypt(GPUBlock m);
  void decrypt(GPUBlock m);
};

#endif
