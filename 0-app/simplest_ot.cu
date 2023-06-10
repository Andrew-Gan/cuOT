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
  if (role == Sender) {
    senders[id] = nullptr;
    delete aes1;
  }
  else
    recvers[id] = nullptr;
  delete aes0;
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
  A = other->A = pow(g, a);
  while(B == 0);
  uint8_t *k0 = hash(pow(B.load(), a));
  uint8_t *k1 = hash(pow(B.load() / A.load(), a));
  aes0 = new Aes(k0);
  aes1 = new Aes(k1);
  for (int i = 0; i < m0.size(); i++) {
    GPUBlock mp0 = m0.at(i);
    GPUBlock mp1 = m1.at(i);
    aes0->encrypt(mp0);
    aes1->encrypt(mp1);
    other->e[0].push_back(mp0);
    other->e[1].push_back(mp1);
  }
  eReceived = other->eReceived = true;
  delete[] k0;
  delete[] k1;
  EventLog::end(BaseOTSend);
}

std::vector<GPUBlock> SimplestOT::recv(uint64_t choices) {
  EventLog::start(BaseOTRecv);
  uint8_t b = rand() % 32;
  while (A == 0);
  B = pow(g, b);
  if (c == 1)
    B = B * A;
  other->B.store(B);
  while(other->A == 0);
  uint8_t *kb = hash(pow(A.load(), b));
  aes0 = new Aes(kb);
  delete[] kb;
  while(!eReceived);
  std::vector<GPUBlock> res;
  for (int i = 0; i < e[0].size(); i++) {
    uint8_t choice = choices & (1 << i) >> i;
    aes0->decrypt(e[choice].at(i));
    res.push_back(e[choice].at(i));
  }
  eReceived = other->eReceived = false;
  EventLog::end(BaseOTRecv);
  return res;
}
