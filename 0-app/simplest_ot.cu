#include "simplest_ot.h"

SimplestOT::SimplestOT(Role role, int id) : OT(role, id) {
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, 1234ULL);
  if (role == Sender) {
    while(recvers[id] == nullptr);
    OT *recv = recvers[id];
    other = dynamic_cast<SimplestOT*>(recv);
  }
  else {
    while(senders[id] == nullptr);
    OT *send = senders[id];
    other = dynamic_cast<SimplestOT*>(send);
  }
}

SimplestOT::~SimplestOT() {
  curandDestroyGenerator(prng);
  if (role == Sender)
    senders[id] = nullptr;
  else
    recvers[id] = nullptr;
}

uint8_t* SimplestOT::hash(uint64_t m) {
  uint8_t *key = new uint8_t[16];
  for (int i = 0; i < 16; i++) {
    key[i] = (m >> i * 4) & 0xff;
  }
  return key;
}

void SimplestOT::send(std::vector<GPUBlock> &m0, std::vector<GPUBlock> &m1) {
  EventLog::start(BaseOTSend);
  while(eReceived);
  uint8_t a = rand() % 32;
  count = other->count = m0.size();
  A = other->A = pow(g, a);
  while(B.size() < count);
  other->e[0].clear();
  other->e[1].clear();
  printf("send count %lu\n", count.load());
  for (int i = 0; i < count; i++) {
    uint8_t *k0 = hash(pow(B.at(i), a));
    uint8_t *k1 = hash(pow(B.at(i) / A, a));
    aes0 = Aes(k0);
    aes1 = Aes(k1);
    GPUBlock mp0 = m0.at(i);
    GPUBlock mp1 = m1.at(i);
    aes0.encrypt(mp0);
    aes1.encrypt(mp1);
    other->e[0].push_back(mp0);
    other->e[1].push_back(mp1);
    delete[] k0;
    delete[] k1;
  }
  eReceived = other->eReceived = true;
  EventLog::end(BaseOTSend);
}

std::vector<GPUBlock> SimplestOT::recv(uint64_t c) {
  EventLog::start(BaseOTRecv);
  std::vector<GPUBlock> res;
  uint8_t b = rand() % 32;
  while (A == 0);
  uint64_t b0 = pow(g, b);
  uint64_t b1 = b0 * A;
  for (int i = 0; i < count; i++) {
    uint8_t choice = c & (1 << i) >> i;
    other->B.push_back(choice == 0 ? b0 : b1);
  }
  uint8_t *kb = hash(pow(A.load(), b));
  printf("recv count\n");
  aes0 = Aes(kb);
  delete[] kb;
  while(!eReceived);
  for (int i = 0; i < count; i++) {
    uint8_t choice = c & (1 << i) >> i;
    aes0.decrypt(e[choice][i]);
    res.push_back(e[choice][i]);
  }
  eReceived = other->eReceived = false;
  EventLog::end(BaseOTRecv);
  return res;
}
