#ifndef __SIMPLEST_OT_H__
#define __SIMPLEST_OT_H__

#include <curand.h>
#include <atomic>
#include <vector>
#include "gpu_block.h"
#include <array>
#include "cryptoTools/Crypto/SodiumCurve.h"
#include "cryptoTools/Crypto/PRNG.h"

using Point = osuCrypto::Sodium::Rist25519;
using Number = osuCrypto::Sodium::Prime25519;

class SimplestOT {
public:
  enum Role { Sender, Recver };
  SimplestOT(Role role, int id, uint64_t count);
  virtual ~SimplestOT();
  std::array<GPUBlock, 2> send();
  GPUBlock recv(uint64_t choice);

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

extern std::array<std::atomic<SimplestOT*>, 100> simplestOTSenders;
extern std::array<std::atomic<SimplestOT*>, 100> simplestOTRecvers;

#endif
