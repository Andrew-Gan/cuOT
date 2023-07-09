#ifndef __AES_H__
#define __AES_H__

#include <curand_kernel.h>
#include "util.h"
#include "gpu_data.h"

class Aes {
private:
  curandGenerator_t prng;
  uint8_t *encExpKey_d = nullptr;
  uint8_t *decExpKey_d = nullptr;

public:
  Aes() {}
  void init(uint8_t *newkey);
  virtual ~Aes();
  static void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  static void expand_decKey(uint8_t *decExpKey, uint8_t *key);
  void decrypt(GPUdata &msg);
  void encrypt(GPUdata &msg);
  void expand_async(OTblock *output_d, GPUdata &m, OTblock *input_d, uint64_t width, int dir, cudaStream_t &s);
};

#endif
