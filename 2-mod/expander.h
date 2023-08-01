#ifndef __AES_H__
#define __AES_H__

#include "util.h"
#include "gpu_data.h"

enum ExpanderType { AesHash_t };

class Expander {
public:
  virtual void expand_async(OTblock *inter, GPUdata &left, GPUdata &right,
    OTblock *input_d, uint64_t width, cudaStream_t &s) = 0;
};

class AesHash : public Expander {
private:
  uint8_t *keyLeft_d = nullptr;
  uint8_t *keyRight_d = nullptr;
  void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  void expand_decKey(uint8_t *decExpKey, uint8_t *key);

public:
  AesHash(uint8_t *newleft, uint8_t *newRight);
  virtual ~AesHash();
  virtual void expand_async(OTblock *inter, GPUdata &left, GPUdata &right,
    OTblock *input_d, uint64_t width, cudaStream_t &s);
  
  // void decrypt(GPUdata &msg);
  // void encrypt(GPUdata &msg);
};

#endif
