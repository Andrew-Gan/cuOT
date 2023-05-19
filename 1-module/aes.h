#ifndef __AES_H__
#define __AES_H__

#include <curand_kernel.h>
#include "util.h"
#include "gpu_block.h"

class Aes {
private:
  curandGenerator_t prng;
  uint8_t *encExpKey_d = nullptr;
  uint8_t *decExpKey_d = nullptr;
  void init();

public:
  uint8_t key[AES_KEYLEN];
  Aes();
  Aes(uint8_t *newkey);
  virtual ~Aes();
  static void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  static void expand_decKey(uint8_t *decExpKey, uint8_t *key);
  void decrypt(GPUBlock &msg);
  void encrypt(GPUBlock &msg);
  void hash_async(TreeNode *output_d, GPUBlock &m, TreeNode *input_d, size_t width, int dir);
};

#endif
