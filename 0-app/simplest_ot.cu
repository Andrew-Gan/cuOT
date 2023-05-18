#include "simplest_ot.h"

SimplestOT::SimplestOT(Role role, int id) : OT(role, id) {
  EventLog::start(BaseOTInit);
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
  EventLog::start(BaseOTInit);
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

void SimplestOT::send(GPUBlock &m0, GPUBlock &m1) {
  EventLog::start(BaseOTSend);
  uint8_t a = rand() % 32;
  A = other->A = pow(g, a);
  while(B == 0);
  uint8_t *k0 = hash(pow(B.load(), a));
  uint8_t *k1 = hash(pow(B.load() / A.load(), a));
  aes0 = new Aes(k0);
  aes1 = new Aes(k1);
  GPUBlock mp0 = m0;
  GPUBlock mp1 = m1;
  aes0->encrypt(mp0);
  aes1->encrypt(mp1);
  other->e[0] = mp0;
  other->e[1] = mp1;
  eReceived = other->eReceived = true;
  delete[] k0;
  delete[] k1;
  while(eReceived);
  EventLog::end(BaseOTSend);
}

GPUBlock SimplestOT::recv(uint8_t c) {
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
  aes0->decrypt(e[c]);
  eReceived = other->eReceived = false;
  EventLog::end(BaseOTRecv);
  return e[c];
}
