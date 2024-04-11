#ifndef __AES_H__
#define __AES_H__

#include "gpu_define.h"
#include "gpu_matrix.h"

struct AES_ctx {
  uint8_t roundKey[11*AES_KEYLEN];
};

enum PprfType { Aes_t };

class Pprf {
public:
  virtual void expand(Mat &interleaved_in, Mat &interleaved_out, Mat &separated, uint64_t inWidth) = 0;
};

class Aes : public Pprf {
private:
  bool hasBothKeys = false;
  uint32_t *keyL, *keyR;
  static void expand_encKey(uint8_t *encExpKey, uint8_t *key);
  static void expand_decKey(uint8_t *decExpKey, uint8_t *key);
  static void single_step(std::vector<uint32_t> &expKey, uint32_t stepIdx);
  static void exp_func(std::vector<uint32_t> &keyArray, std::vector<uint32_t> &expKeyArray);
  static uint32_t galois_prod(uint32_t a, uint32_t b);
  static void inv_mix_col(std::vector<unsigned> &temp);
  static void inv_exp_func(std::vector<unsigned> &expKey, std::vector<unsigned> &invExpKey);

public:
  Aes(void *leftUnexpSeed, void *rightUnexpSeed = nullptr);
  virtual void encrypt(Mat &data);
  virtual void expand(Mat &interleaved_in, Mat &interleaved_out, Mat &separated, uint64_t inWidth);
};

#endif
