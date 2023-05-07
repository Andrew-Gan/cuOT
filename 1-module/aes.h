#ifndef __AES_H__
#define __AES_H__

#include <curand_kernel.h>
#include "util.h"

class AesBlocks {
public:
  AesBlocks();
  AesBlocks(size_t nBlock);
  AesBlocks(const AesBlocks &blk);
  virtual ~AesBlocks();
  uint8_t *data_d = nullptr;
  size_t nBlock = 0;
  AesBlocks operator^(const AesBlocks &d_rhs);
  AesBlocks operator=(const AesBlocks &rhs);
  bool operator==(const AesBlocks &rhs);
  void set(uint32_t rhs);
};

class Aes {
private:
  curandGenerator_t prng;
  uint8_t *encExpKey_d = nullptr;
  uint8_t *decExpKey_d = nullptr;
public:
  uint8_t key[AES_KEYLEN];
  Aes();
  Aes(uint8_t *newkey);
  virtual ~Aes();
  static void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  static void expand_decKey(uint8_t *decExpKey, uint8_t *key);
  void decrypt(AesBlocks &msg);
  void encrypt(AesBlocks &msg);
};

#endif
