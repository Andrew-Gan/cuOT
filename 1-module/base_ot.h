#ifndef __BASE_OT_H__
#define __BASE_OT_H__

#include <atomic>
#include "util.h"
#include "rsa.h"
#include "aes.h"

enum Role { Sender, Recver };
enum InitStatus {noInit, rsaInitDone, xInitDone};
enum OTStatus {notReady, vReady, mReady};

class BaseOT {
private:
  // network shared
  uint64_t e = 0, n = 0;
  uint8_t aesKey_enc[16] = {};
  GPUBlock x[2], v;
  GPUBlock mp[2];
  // sender only
  GPUBlock k0, k1;
  curandGenerator_t prng_s;
  // recver only
  curandGenerator_t prng_r;
  // misc
  int id = -1;
  Rsa *rsa;
  Role role;
  BaseOT *other;
  void sender_init(int id);
  void recver_init(int id);

public:
  BaseOT(Role role, int id);
  virtual ~BaseOT();
  void send(GPUBlock &m0, GPUBlock &m1);
  GPUBlock recv(uint8_t b);
};

#endif
