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
  delete aes0;
  delete aes1;
}

uint8_t* SimplestOT::hash(uint64_t m) {
  uint8_t *key = new uint8_t[16];
  for (int i = 0; i < 16; i++) {
    key[i] = (m >> i * 4) & 0xff;
  }
  return key;
}

void SimplestOT::send(GPUBlock &m0, GPUBlock &m1) {
  uint8_t a = rand() % 64;
  other->A = pow(g, a);
  while(B == 0);
  uint8_t *k0 = hash(pow(B, a));
  uint8_t *k1 = hash(pow(B / A, a));
  aes0 = new Aes(k0);
  aes1 = new Aes(k1);
  aes0->encrypt(m0);
  aes1->encrypt(m1);
  other->e[0] = m0;
  other->e[1] = m1;
  other->eReceived = true;
  delete[] k0;
  delete[] k1;
}

GPUBlock SimplestOT::recv(uint8_t c) {
  uint8_t b = rand() % 64;
  while (A == 0);
  B = pow(g, b);
  if (c == 1)
    B *= A;
  other->B = B;
  while(other->A == 0);
  uint8_t *kb = hash(pow(A, b));
  aes0 = new Aes(kb);
  delete[] kb;
  while(!eReceived);
  aes0->decrypt(e[c]);
  return e[c];
}
