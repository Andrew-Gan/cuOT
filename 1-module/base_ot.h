#ifndef __BASE_OT_H__
#define __BASE_OT_H__

#include <atomic>
#include "util.h"
#include "rsa.h"
#include "aes.h"

enum Role { Sender, Recver };
enum InitStatus {noInit, rsaInitDone, aesInitDone};
enum OTStatus {notReady, vReady, mReady};

class BaseOT {
private:
  // network shared
  uint64_t e = 0, n = 0;
  uint8_t aesKey_enc[16] = {};
  AesBlocks x[2], v;
  AesBlocks mp[2];
  std::atomic<InitStatus> initStatus;
  std::atomic<OTStatus> otStatus;
  // sender only
  AesBlocks k0, k1;
  curandGenerator_t prng_s;
  // recver only
  curandGenerator_t prng_r;
  // misc
  Rsa *rsa;
  Aes *aes = nullptr;
  Role role;
  BaseOT *other;
  void sender_init(int id);
  void recver_init(int id);

public:
  BaseOT(Role role, int id);
  virtual ~BaseOT();
  void send(AesBlocks &m0, AesBlocks &m1);
  AesBlocks recv(uint8_t b);
};

#endif
