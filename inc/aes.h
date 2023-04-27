#ifndef __AES_H__
#define __AES_H__

#include <curand_kernel.h>
#include "util.h"

void aes_init_ctx(AES_ctx* ctx, const uint8_t* key);

class AesBlocks {
public:
  AesBlocks();
  AesBlocks(size_t nBlock);
  virtual ~AesBlocks();
  uint8_t *d_data = nullptr;
  size_t nBlock = 0;
  AesBlocks operator^(const AesBlocks &d_rhs);
  AesBlocks operator=(uint32_t rhs);
  AesBlocks operator=(const AesBlocks &rhs);
};

class Aes {
private:
  curandGenerator_t prng;
public:
  uint8_t *d_key = nullptr;
  Aes();
  Aes(uint8_t *newkey);
  virtual ~Aes();
  void decrypt(AesBlocks msg);
  void encrypt(AesBlocks msg);
};

#endif
