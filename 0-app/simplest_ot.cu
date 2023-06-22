#include "simplest_ot.h"
#include "cryptoTools/Crypto/RandomOracle.h"
#include "cryptoTools/Crypto/SodiumCurve.h"

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

  hasContent[0] = false;
  hasContent[1] = false;
}

SimplestOT::~SimplestOT() {
  if (role == Sender)
    simplestOTSenders[id] = nullptr;
  else
    simplestOTRecvers[id] = nullptr;
}

void SimplestOT::fromOwnBuffer(uint8_t *d, int id, size_t nBytes) {
  while (!hasContent[id]);
  memcpy(d, buffer[id], nBytes);
  hasContent[id] = false;
}

void SimplestOT::toOtherBuffer(uint8_t *s, int id, size_t nBytes) {
  while (other->hasContent[id]);
  memcpy(other->buffer[id], s, nBytes);
  other->hasContent[id] = true;
}

std::array<std::vector<GPUBlock>, 2> SimplestOT::send(size_t count) {
  EventLog::start(BaseOTSend);
  uint64_t a = rand() & ((1 << 5) - 1);
  A = a;
  osuCrypto::Sodium::Rist25519::mulGenerator(A);
  n = count;
  toOtherBuffer((uint8_t*) &A, 0, sizeof(A));

  B.resize(n);
  fromOwnBuffer((uint8_t*) &B.at(0), 0, sizeof(B.at(0)) * B.size());

  std::array<std::vector<GPUBlock>, 2> m;
  m[0] = std::vector<GPUBlock>(n, GPUBlock(TREENODE_SIZE));
  m[1] = std::vector<GPUBlock>(n, GPUBlock(TREENODE_SIZE));
  A *= a;

  for (size_t i = 0; i < n; i++) {
    B.at(i) *= a;
    osuCrypto::RandomOracle ro(TREENODE_SIZE);
    printf("send0: i = %lu, B = %lu\n", i, B.at(i));
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff0[TREENODE_SIZE];
    ro.Final(buff0);
    cudaMemcpy(m[0].at(i).data_d, buff0, TREENODE_SIZE, cudaMemcpyHostToDevice);

    B.at(i) -= A;
    ro.Reset();
    printf("send1: i = %lu, B = %lu\n", i, B.at(i));
    ro.Update(B.at(i));
    ro.Update(i);
    uint8_t buff1[TREENODE_SIZE];
    ro.Final(buff1);
    cudaMemcpy(m[1].at(i).data_d, buff1, TREENODE_SIZE, cudaMemcpyHostToDevice);
  }
  EventLog::end(BaseOTSend);
  return m;
}

std::vector<GPUBlock> SimplestOT::recv(size_t count, uint64_t choice) {
  fromOwnBuffer((uint8_t*) &A, 0, sizeof(A));
  n = count;
  EventLog::start(BaseOTRecv);
  std::vector<GPUBlock> mb(n, TREENODE_SIZE);
  std::vector<uint64_t> b(n);
  for (size_t i = 0; i < n; i++) {
    b.at(i) = rand() & ((1 << 5) - 1);
    uint64_t B0 = b.at(i);
    osuCrypto::Sodium::Rist25519::mulGenerator(B0);
    uint64_t B1 = A + B0;
    uint8_t c = choice & (1 << i) >> i;
    B.push_back(c == 0 ? B0 : B1);
  }
  toOtherBuffer((uint8_t*) &B.at(0), 0, sizeof(B.at(0)) * B.size());
  uint8_t buff[TREENODE_SIZE];
  for (size_t i = 0; i < n; i++) {
    uint64_t mB = A * b.at(i);
    osuCrypto::RandomOracle ro(TREENODE_SIZE);
    printf("recv: i = %lu, B = %lu\n", i, mB);
    ro.Update(mB);
    ro.Update(i);
    ro.Final(buff);
    cudaMemcpy(mb.at(i).data_d, buff, TREENODE_SIZE, cudaMemcpyHostToDevice);
  }
  EventLog::end(BaseOTRecv);
  return mb;
}
