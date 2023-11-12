#ifndef __SIMPLEST_OT_H__
#define __SIMPLEST_OT_H__

#include <atomic>
#include <vector>
#include "gpu_vector.h"
#include <array>
#include "cryptoTools/Crypto/SodiumCurve.h"
#include "cryptoTools/Crypto/PRNG.h"

using Point = osuCrypto::Sodium::Rist25519;
using Number = osuCrypto::Sodium::Prime25519;

enum BaseOTType { SimplestOT_t };

class OT {
public:
  virtual std::array<vec, 2> send() = 0;
  virtual vec recv(uint64_t choice) = 0;
};

class SimplestOT : public OT {
public:
  SimplestOT(Role role, int id, uint64_t count);
  virtual ~SimplestOT();
  virtual std::array<vec, 2> send();
  virtual vec recv(uint64_t choice);

private:
  Role mRole;
  int mID;
  uint64_t mCount = 0;
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

extern std::array<std::atomic<OT*>, 100> baseOTSenders;
extern std::array<std::atomic<OT*>, 100> baseOTRecvers;

#endif
