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
  AesBlocks d_mp[2];
  std::atomic<InitStatus> initStatus;
  std::atomic<OTStatus> otStatus;
  // sender only
  AesBlocks d_k0, d_k1;
  curandGenerator_t prng_s;
  // recver only
  curandGenerator_t prng_r;
  // misc
  Rsa rsa;
  Aes aes;
  Role role;
  BaseOT *other;
  void senderInit(int id);
  void recverInit(int id);

public:
  BaseOT(Role role, int id);
  void ot_send(AesBlocks d_m0, AesBlocks d_m1);
  AesBlocks ot_recv(uint8_t b, size_t nBytes);
};

#endif
