#ifndef __AES_H__
#define __AES_H__

#include "gpu_define.h"
#include "gpu_matrix.h"
#include "gpu_span.h"

struct AES_ctx {
  uint8_t roundKey[11*AES_KEYLEN];
};

enum ExpandType { AesExpand_t };

class Expand {
public:
  virtual void expand(Mat &interleaved_in, Mat &interleaved_out, Mat &separated, uint64_t inWidth) = 0;
};

class AesExpand : public Expand {
private:
  uint32_t *keyL, *keyR;
  void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  void expand_decKey(uint8_t *decExpKey, uint8_t *key);

public:
  AesExpand(void *leftUnexp, void *rightUnexp);
  virtual void expand(Span &interleaved_in, Span &interleaved_out, Mat &separated, uint64_t inWidth);
  virtual void expand(Mat &interleaved_in, Mat &interleaved_out, Mat &separated, uint64_t inWidth);
};

#endif
