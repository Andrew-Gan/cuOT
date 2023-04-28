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
  static inline uint64_t e, n;
  static inline uint8_t aesKey_enc[16];
  static inline AesBlocks x[2], v;
  static inline AesBlocks d_mp[2];
  static inline InitStatus initStatus = noInit;
  static inline OTStatus otStatus = notReady;
  // sender only
  AesBlocks d_k0, d_k1;
  curandGenerator_t prng_s;
  // recver only
  curandGenerator_t prng_r;
  // misc
  Rsa rsa;
  Aes aes;
  Role role;

public:
  BaseOT(Role role, size_t msgSize);
  void ot_send(AesBlocks d_m0, AesBlocks d_m1);
  AesBlocks ot_recv(uint8_t b, size_t nBytes);
};

#endif
