#ifndef __AES_H__
#define __AES_H__

#include "gpu_tools.h"
#include "gpu_vector.h"

struct AES_ctx {
  uint8_t roundKey[11*AES_KEYLEN];
};

enum ExpandType { AesExpand_t };

class Expand {
public:
  virtual void expand(Vec &interleaved, Vec &separated, uint64_t inWidth) = 0;
};

class AesExpand : public Expand {
private:
  void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  void expand_decKey(uint8_t *decExpKey, uint8_t *key);

public:
  AesExpand(uint8_t *leftUnexp, uint8_t *rightUnexp);
  virtual void expand(Span &interleaved, Vec &separated, uint64_t inWidth);
  virtual void expand(Vec &interleaved, Vec &separated, uint64_t inWidth);
};

#endif
