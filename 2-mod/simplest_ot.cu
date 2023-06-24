#include "simplest_ot.h"
#include "cryptoTools/Crypto/RandomOracle.h"
#include <ctime>

std::array<std::atomic<SimplestOT*>, 100> simplestOTSenders;
std::array<std::atomic<SimplestOT*>, 100> simplestOTRecvers;

SimplestOT::SimplestOT(Role myrole, int myid) : role(myrole), id(myid) {
  if (role == Sender) {
    simplestOTSenders[id] = this;
    while(simplestOTRecvers[id] == nullptr);
    other = simplestOTRecvers[id];
  }
  else {
    simplestOTRecvers[id] = this;
    while(simplestOTSenders[id] == nullptr);
    other = simplestOTSenders[id];
  }
  hasContent = false;
  prng.SetSeed(osuCrypto::block(clock(), 0));
}

SimplestOT::~SimplestOT() {
  if (role == Sender)
    simplestOTSenders[id] = nullptr;
  else
    simplestOTRecvers[id] = nullptr;
}

void SimplestOT::fromOwnBuffer(uint8_t *d, size_t nBytes) {
  while (!hasContent);
  memcpy(d, buffer, nBytes);
  hasContent = false;
}

void SimplestOT::toOtherBuffer(uint8_t *s, size_t nBytes) {
  while (other->hasContent);
  memcpy(other->buffer, s, nBytes);
  other->hasContent = true;
}

std::array<std::vector<GPUBlock>, 2> SimplestOT::send(size_t count) {
  a.randomize(prng);
  A = Point::mulGenerator(a);
  toOtherBuffer((uint8_t*) &A, sizeof(A));

  B.resize(count);
  std::array<std::vector<GPUBlock>, 2> m;
  m[0] = std::vector<GPUBlock>(count, GPUBlock(BLK_SIZE));
  m[1] = std::vector<GPUBlock>(count, GPUBlock(BLK_SIZE));
  A *= a;
  fromOwnBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  for (size_t i = 0; i < count; i++) {
    B.at(i) *= a;
    osuCrypto::RandomOracle ro(BLK_SIZE);
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff0[BLK_SIZE];
    ro.Final(buff0);
    cudaMemcpy(m[0].at(i).data_d, buff0, BLK_SIZE, cudaMemcpyHostToDevice);
    B.at(i) -= A;
    ro.Reset();
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff1[BLK_SIZE];
    ro.Final(buff1);
    cudaMemcpy(m[1].at(i).data_d, buff1, BLK_SIZE, cudaMemcpyHostToDevice);
  }
  return m;
}

std::vector<GPUBlock> SimplestOT::recv(size_t count, uint64_t choice) {
  fromOwnBuffer((uint8_t*) &A, sizeof(A));
  std::vector<GPUBlock> mb(count, BLK_SIZE);
  for (size_t i = 0; i < count; i++) {
    b.emplace_back(prng);
    Point B0 = Point::mulGenerator(b.at(i));
    Point B1 = A + B0;
    uint64_t c = choice & (1 << i);
    B.push_back(c == 0 ? B0 : B1);
  }
  toOtherBuffer((uint8_t*) &B.at(0), sizeof(B.at(0)) * B.size());

  uint8_t buff[BLK_SIZE];
  for (size_t i = 0; i < count; i++) {
    Point mB = A * b.at(i);
    osuCrypto::RandomOracle ro(BLK_SIZE);
    ro.Update(mB);
    ro.Update(i);
    ro.Final(buff);
    cudaMemcpy(mb.at(i).data_d, buff, BLK_SIZE, cudaMemcpyHostToDevice);
  }
  return mb;
}
