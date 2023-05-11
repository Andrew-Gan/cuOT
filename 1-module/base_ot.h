#ifndef __BASE_OT_H__
#define __BASE_OT_H__

#include <atomic>
#include <curand.h>
#include "util.h"
#include "rsa.h"
#include "gpu_block.h"

enum Role { Sender, Recver };
enum InitStatus {noInit, rsaInitDone, xInitDone};
enum OTStatus {notReady, vReady, mReady};

class BaseOT {
private:
  // network shared - common
  uint64_t e, n;
  GPUBlock x[2];
  // network shared - unique
  GPUBlock v, mp[2];
  // sender - not shared
  GPUBlock k[2];
  curandGenerator_t prng_s;
  // recver - not shared
  curandGenerator_t prng_r;
  // misc
  Role role;
  int id;
  Rsa *rsa;
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
