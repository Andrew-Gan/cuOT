#ifndef __AES_H__
#define __AES_H__

#include "gpu_tools.h"
#include "gpu_vector.h"

enum ExpandType { AesHash_t };

class Expand {
public:
  virtual void expand(vec &interleaved, vec &separated, vec &input, uint64_t width) = 0;
};

class AesHash : public Expand {
private:
  uint8_t *keyLeft_d = nullptr;
  uint8_t *keyRight_d = nullptr;
  void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  void expand_decKey(uint8_t *decExpKey, uint8_t *key);

public:
  AesHash(uint8_t *newleft, uint8_t *newRight);
  virtual ~AesHash();
  virtual void expand(blk *interleaved, vec &separated, blk *input, uint64_t width);
  virtual void expand(vec &interleaved, vec &separated, vec &input, uint64_t width);
};

#endif
