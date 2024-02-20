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

std::array<Mat, 2> SimplestOT::send() {
  a.randomize(prng);
  A = Point::mulGenerator(a);
  toOtherBuffer((uint8_t*) &A, sizeof(A));

  std::array<Mat, 2> m = {
    Mat({mCount}), Mat({mCount})
  };
  A *= a;
  fromOwnBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  blk buff0[mCount];
  blk buff1[mCount];

  for (uint64_t i = 0; i < mCount; i++) {
    B.at(i) *= a;
    osuCrypto::RandomOracle ro(sizeof(blk));
    ro.Update(B.at(i));
    ro.Update(i);
    ro.Final((uint8_t*)(buff0 + i));
    B.at(i) -= A;
    ro.Reset();
    ro.Update(B.at(i));
    ro.Update(i);
    ro.Final((uint8_t*)(buff1 + i));
  }
  cudaMemcpy(m[0].data(), buff0, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  cudaMemcpy(m[1].data(), buff1, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  return m;
}

Mat SimplestOT::recv(uint64_t choice) {
  fromOwnBuffer((uint8_t*) &A, sizeof(A));
  Mat mb({mCount});

  for (uint64_t i = 0; i < mCount; i++) {
    b.emplace_back(prng);
    Point B0 = Point::mulGenerator(b.at(i));
    Point B1 = A + B0;
    uint64_t c = choice & (1 << i);
    B.at(i) = c == 0 ? B0 : B1;
  }
  toOtherBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  blk buff[mCount];
  for (uint64_t i = 0; i < mCount; i++) {
    Point point = A * b.at(i);
    osuCrypto::RandomOracle ro(sizeof(blk));
    ro.Update(point);
    ro.Update(i);
    ro.Final((uint8_t*)(buff + i));
  }
  cudaMemcpy(mb.data(), buff, mCount * sizeof(blk), cudaMemcpyHostToDevice);
  return mb;
}
