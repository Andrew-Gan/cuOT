#ifndef __BASE_OT_H__
#define __BASE_OT_H__

#include <atomic>
#include "util.h"

enum Role {Sender, Recver};
enum InitStatus {noInit, rsaInitDone, aesInitDone};
enum OTStatus {notReady, vReady, mReady};

__global__
void xor_gpu(uint32_t *out, uint32_t *data, uint32_t val);

class BaseOT {
private:
  // network shared
  static uint32_t e, n, x[2], v, aesKey_enc;
  static uint32_t* m[2];
  static InitStatus initStatus;
  static OTStatus otStatus;
  // sender
  uint32_t d, k_s[2], aesKey_s;
  // recver
  uint32_t k_r = 0, aesKey_r;
  // misc
  Role role;
  void rsa_init();
  uint32_t aesDecrypt(uint32_t m, uint32_t d, uint32_t n);
  uint32_t aesEncrypt(uint32_t m, uint32_t e, uint32_t n);

public:
  BaseOT(Role role);
  void ot_send(uint8_t *m0, uint8_t *m1, size_t nBytes);
  uint8_t* ot_recv(uint8_t b, size_t nBytes);
};

#endif
