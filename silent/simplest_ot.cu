#include "base_ot.h"
#include "cryptoTools/Crypto/RandomOracle.h"
#include <ctime>

std::array<std::atomic<SimplestOT*>, 100> simplestOTSenders;
std::array<std::atomic<SimplestOT*>, 100> simplestOTRecvers;

SimplestOT::SimplestOT(Role role, int id, uint64_t count) :
  mRole(role), mID(id), mCount(count) {

  hasContent = false;
  prng.SetSeed(osuCrypto::block(clock(), 0));
  buffer = new uint8_t[mCount * sizeof(Point)];
  B.resize(mCount);

  if (mRole == Sender) {
    simplestOTSenders[mID] = this;
    while(simplestOTRecvers[mID] == nullptr);
    other = simplestOTRecvers[mID];
  }
  else {
    simplestOTRecvers[mID] = this;
    while(simplestOTSenders[mID] == nullptr);
    other = simplestOTSenders[mID];
  }
}

SimplestOT::~SimplestOT() {
  if (mRole == Sender)
    simplestOTSenders[mID] = nullptr;
  else
    simplestOTRecvers[mID] = nullptr;
  delete[] buffer;
}

void SimplestOT::fromOwnBuffer(uint8_t *d, uint64_t nBytes) {
  while (!hasContent);
  memcpy(d, buffer, nBytes);
  hasContent = false;
}

void SimplestOT::toOtherBuffer(uint8_t *s, uint64_t nBytes) {
  while (other->hasContent);
  memcpy(other->buffer, s, nBytes);
  other->hasContent = true;
}

std::array<vec, 2> SimplestOT::send() {
  a.randomize(prng);
  A = Point::mulGenerator(a);
  toOtherBuffer((uint8_t*) &A, sizeof(A));

  std::array<vec, 2> m = {
    vec(mCount), vec(mCount)
  };
  A *= a;
  fromOwnBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  blk buff0[mCount * sizeof(blk)];
  blk buff1[mCount * sizeof(blk)];

  for (uint64_t i = 0; i < mCount; i++) {
    B.at(i) *= a;
    osuCrypto::RandomOracle ro(sizeof(blk));
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff0[sizeof(blk)];
    ro.Final(buff0 + i);
    B.at(i) -= A;
    ro.Reset();
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff1[sizeof(blk)];
    ro.Final(buff1 + i);
  }
  cudaMemcpy(m[0].data(), buff0, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  cudaMemcpy(m[1].data(), buff1, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  return m;
}

vec SimplestOT::recv(uint64_t choice) {
  fromOwnBuffer((uint8_t*) &A, sizeof(A));
  vec mb(mCount);

  for (uint64_t i = 0; i < mCount; i++) {
    b.emplace_back(prng);
    Point B0 = Point::mulGenerator(b.at(i));
    Point B1 = A + B0;
    uint64_t c = choice & (1 << i);
    B.at(i) = c == 0 ? B0 : B1;
  }
  toOtherBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  uint8_t buff[mCount * sizeof(blk)];

  for (uint64_t i = 0; i < mCount; i++) {
    Point point = A * b.at(i);
    osuCrypto::RandomOracle ro(sizeof(blk));
    ro.Update(point);
    ro.Update(i);
    ro.Final(buff);
  }
  cudaMemcpy(mb.data(), buff, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  return mb;
}
