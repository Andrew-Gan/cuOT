#ifndef __SIMPLEST_OT_H__
#define __SIMPLEST_OT_H__

#include "gpu_matrix.h"
#include <atomic>
#include <vector>
#include <array>
#include "cryptoTools/Crypto/SodiumCurve.h"
#include "cryptoTools/Crypto/PRNG.h"

using Point = osuCrypto::Sodium::Rist25519;
using Number = osuCrypto::Sodium::Prime25519;

enum BaseOTType { SimplestOT_t };

class OT {
public:
  Role mRole;
  int mID;
  uint64_t mCount;
  virtual void send(blk *m0, blk *m1) = 0;
  virtual void recv(blk *mb, uint64_t choice) = 0;

  OT(Role role, int id, uint64_t count) : mRole(role), mID(id), mCount(count) {}
};

class SimplestOT : public OT {
public:
  SimplestOT(Role role, int id, uint64_t count);
  virtual ~SimplestOT();
  virtual void send(blk *m0, blk *m1);
  virtual void recv(blk *mb, uint64_t choice);

private:
  Number a;
  std::vector<Number> b;
  Point A;
  std::vector<Point> B;
  osuCrypto::PRNG prng;
  SimplestOT *other = nullptr;
  uint8_t *buffer = nullptr;
  std::atomic<bool> hasContent;

  void fromOwnBuffer(uint8_t *d, uint64_t nBytes);
  void toOtherBuffer(uint8_t *s, uint64_t nBytes);
};

#endif
